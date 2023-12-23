#include "polyscope/polyscope.h"

#include <igl/PI.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/boundary_loop.h>
#include <igl/exact_geodesic.h>
#include <igl/gaussian_curvature.h>
#include <igl/invert_diag.h>
#include <igl/lscm.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

#include <iostream>
#include <unordered_set>
#include <utility>
#include <cmath>

#include <igl/triangle/triangulate.h>
#include <sstream>
#include <igl/writeOBJ.h>
#include <misc/cpp/imgui_stdlib.h>

constexpr double PI = 3.1415926535898;

double potential(Eigen::Vector2d p, bool wrap)
{
    double r = p.norm();
    double theta = std::atan2(p[1], p[0]);
    if (wrap && theta < 0)
        theta += 2.0 * PI;
    return 1.0 / 3.0 * r * r * (-9.0 * std::sin(theta / 2.0) + 5.0 * std::sin(3.0 * theta / 2.0));
}

void makeDisk(double triArea, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd &VF, double innerRadius, bool annulus)
{
    double seglen = std::sqrt(4.0 * triArea / std::sqrt(3.0));
    int nsegs = std::max(3, int(2.0 * PI / seglen));
    int innersegs = annulus ? std::max(3, int(2.0 * PI * innerRadius / seglen)) : 0;

    int totsegs = nsegs + innersegs;

    Eigen::MatrixXd inV(totsegs, 2);
    Eigen::MatrixXd inE(totsegs, 2);
    for (int i = 0; i < nsegs; i++)
    {
        inV(i, 0) = std::cos(2.0 * PI * i / nsegs);
        inV(i, 1) = std::sin(2.0 * PI * i / nsegs);
        inE(i, 0) = i;
        inE(i, 1) = (i + 1) % nsegs;
    }
    for (int i = 0; i < innersegs; i++)
    {
        inV(nsegs + i, 0) = innerRadius * std::cos(2.0 * PI * i / innersegs);
        inV(nsegs + i, 1) = innerRadius * std::sin(2.0 * PI * i / innersegs);
        inE(nsegs + i, 0) = nsegs + i;
        inE(nsegs + i, 1) = nsegs + ((i + 1) % innersegs);
    }
    Eigen::MatrixXd inH(0, 2);
    if (annulus)
    {
        inH.resize(1, 2);
        inH << 0, 0;
    }
    std::stringstream ss;
    ss << "a" << std::setprecision(30) << std::fixed << triArea << "qDY";
    Eigen::MatrixXd outV;    
    igl::triangle::triangulate(inV, inE, inH, ss.str(), outV, F);
    V.resize(outV.rows(), 3);
    V.col(0) = outV.col(0);
    V.col(1) = outV.col(1);

    int nfaces = F.rows();
    VF.resize(nfaces, 2);
    for (int i = 0; i < nfaces; i++)
    {
        double pots[3];
        bool hasneg = false;
        bool haspos = false;
        for (int j = 0; j < 3; j++)
        {
            double theta = std::atan2(outV(F(i, j), 1), outV(F(i, j), 1));
            if (theta < 0)
                hasneg = true;
            else if (theta > 0)
                haspos = true;
        }
        bool wrap = (hasneg && haspos);
        for (int j = 0; j < 3; j++)
            pots[j] = potential(outV.row(F(i, j)).transpose(), wrap);
        Eigen::Matrix2d B;
        B.row(0) = outV.row(F(i, 1)).transpose() - outV.row(F(i, 0)).transpose();
        B.row(1) = outV.row(F(i, 2)).transpose() - outV.row(F(i, 0)).transpose();
        Eigen::Vector2d w(pots[1] - pots[0], pots[2] - pots[0]);
        VF.row(i) = (B.inverse() * w).transpose();
    }
}

static bool serializeMatrix(const Eigen::MatrixXd& mat, const std::string& filepath, int vector_per_element) {
    try {
        std::ofstream outFile(filepath, std::ios::binary);
        if (!outFile.is_open()) {
            std::cerr << "Error: Unable to open file for writing: " << filepath << std::endl;
            return false;
        }

        // Write matrix rows and cols
        int rows = static_cast<int>(mat.rows());
        int cols = static_cast<int>(mat.cols());
        int vpe = static_cast<int>(vector_per_element);


        outFile.write("FRA 2", sizeof("FRA 2"));
        outFile.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        outFile.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        outFile.write(reinterpret_cast<const char*>(&vpe), sizeof(int));

        // Write matrix data
        outFile.write(reinterpret_cast<const char*>(mat.data()), rows * cols * sizeof(double));
        outFile.close();
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: Unable to serialize matrix: " << e.what() << std::endl;
        return false;
    }
}

void exportExample(const std::string& filename, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& VF)
{
    std::string objname = filename + ".obj";
    Eigen::MatrixXd V3d(V.rows(), 3);
    V3d.col(0) = V.col(0);
    V3d.col(1) = V.col(1);
    V3d.col(2).setZero();
    igl::writeOBJ(objname, V3d, F);

    std::string bfraname = filename + ".bfra";

    serializeMatrix(VF, bfraname, 1);
    /*std::ofstream ofs(bfraname);
    ofs << "FRA 2" << std::endl;
    ofs << F.rows() << " " << 2 << " " << 1 << std::endl;
    for (int i = 0; i < F.rows(); i++)
    {
        ofs << VF(i, 0) << ", " << VF(i, 1) << std::endl;
    }*/
}

namespace ImGui
{

    struct InputTextCallback_UserData
    {
        std::string* Str;
        ImGuiInputTextCallback  ChainCallback;
        void* ChainCallbackUserData;
    };

    static int InputTextCallback(ImGuiInputTextCallbackData* data)
    {
        InputTextCallback_UserData* user_data = (InputTextCallback_UserData*)data->UserData;
        if (data->EventFlag == ImGuiInputTextFlags_CallbackResize)
        {
            // Resize string callback
            // If for some reason we refuse the new length (BufTextLen) and/or capacity (BufSize) we need to set them back to what we want.
            std::string* str = user_data->Str;
            IM_ASSERT(data->Buf == str->c_str());
            str->resize(data->BufTextLen);
            data->Buf = (char*)str->c_str();
        }
        else if (user_data->ChainCallback)
        {
            // Forward to user callback, if any
            data->UserData = user_data->ChainCallbackUserData;
            return user_data->ChainCallback(data);
        }
        return 0;
    }

    bool ImGui::InputText(const char* label, std::string* str, ImGuiInputTextFlags flags, ImGuiInputTextCallback callback, void* user_data)
    {
        IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
        flags |= ImGuiInputTextFlags_CallbackResize;

        InputTextCallback_UserData cb_user_data;
        cb_user_data.Str = str;
        cb_user_data.ChainCallback = callback;
        cb_user_data.ChainCallbackUserData = user_data;
        return InputText(label, (char*)str->c_str(), str->capacity() + 1, flags, InputTextCallback, &cb_user_data);
    }
};

int main(int argc, char** argv) {

    // Options
    polyscope::options::autocenterStructures = true;
    polyscope::view::windowWidth = 1024;
    polyscope::view::windowHeight = 1024;

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd VF;

    // Initialize polyscope
    polyscope::init();
    polyscope::view::style = polyscope::view::NavigateStyle::Planar;

    double triArea = 0.01;
    double innerRadius = 0.1;

    makeDisk(triArea, V, F, VF, innerRadius, false);
    auto dmesh = polyscope::registerSurfaceMesh2D("Disk", V, F);
    dmesh->edgeWidth = 1.0;
    auto vf = dmesh->addFaceVectorQuantity2D("VF", VF);
    vf->setEnabled(true);

    bool annulus = false;

    std::string filename = "disk";

    polyscope::state::userCallback = [&]()
    {
        ImGui::PushItemWidth(100);

        ImGui::InputDouble("Triangle Area", &triArea);
        ImGui::InputDouble("Inner Radius", &innerRadius);
        ImGui::SameLine();
        ImGui::Checkbox("Make Annulus", &annulus);

        if (ImGui::Button("Recreate Mesh")) {
            makeDisk(triArea, V, F, VF, innerRadius, annulus);
            dmesh = polyscope::registerSurfaceMesh2D("Disk", V, F);
            dmesh->edgeWidth = 1.0;
            vf = dmesh->addFaceVectorQuantity2D("VF", VF);
            vf->setEnabled(true);
        }

        ImGui::InputText("Base Name", &filename);
        if (ImGui::Button("Export Example"))
        {
            exportExample(filename, V, F, VF);
        }

        ImGui::PopItemWidth();
    };

  // Show the gui
  polyscope::show();

  return 0;
}
