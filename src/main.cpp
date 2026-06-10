#include <easy3d/viewer/viewer.h>
#include <easy3d/core/surface_mesh.h>
#include <easy3d/core/types.h>

#include "nested_cages.h"
#include <iostream>
#include <string>
#include <vector>

using namespace easy3d;

// dump easy3d mesh into eigen matrices.
NestedCages::Mesh to_eigen(easy3d::SurfaceMesh *emesh)
{
	NestedCages::Mesh mesh;
	mesh.V.resize(emesh->n_vertices(), 3);
	mesh.F.resize(emesh->n_faces(), 3);

	auto points = emesh->get_vertex_property<vec3>("v:point");

	int v_idx = 0;
	for (auto v : emesh->vertices())
	{
		mesh.V(v_idx, 0) = points[v].x;
		mesh.V(v_idx, 1) = points[v].y;
		mesh.V(v_idx, 2) = points[v].z;
		v_idx++;
	}

	int f_idx = 0;
	for (auto f : emesh->faces())
	{
		int c = 0;
		for (auto v : emesh->vertices(f))
		{
			if (c < 3)
				mesh.F(f_idx, c) = v.idx();
			c++;
		}
		f_idx++;
	}
	return mesh;
}

// convert back to easy3d so the viewer doesn't complain
easy3d::SurfaceMesh *to_easy3d(const NestedCages::Mesh &mesh)
{
	auto emesh = new easy3d::SurfaceMesh();
	std::vector<easy3d::SurfaceMesh::Vertex> vhandles;

	for (int i = 0; i < mesh.V.rows(); ++i)
	{
		vhandles.push_back(emesh->add_vertex(vec3(mesh.V(i, 0), mesh.V(i, 1), mesh.V(i, 2))));
	}

	for (int i = 0; i < mesh.F.rows(); ++i)
	{
		std::vector<easy3d::SurfaceMesh::Vertex> face_v = {
			vhandles[mesh.F(i, 0)],
			vhandles[mesh.F(i, 1)],
			vhandles[mesh.F(i, 2)]};
		emesh->add_face(face_v);
	}
	return emesh;
}

int main(int argc, char **argv)
{
	std::cout << "========================================\n";
	std::cout << "       NESTED CAGES GENERATOR           \n";
	std::cout << "========================================\n\n";

	std::string file_path = "homer.obj";

	// parameters
	NestedCages::Parameters params;
	params.num_layers = 1; // k in the paper
	params.decimation_ratio = 0.63;
	params.flow_dt = 0.001; // flow step size. 1e-3 is from the paper

	std::cout << "Enter mesh file path (default: homer.obj): ";
	std::string input;
	std::getline(std::cin, input);
	if (!input.empty())
		file_path = input;

	std::cout << "Enter number of layers (default: 1): ";
	std::getline(std::cin, input);
	if (!input.empty())
		params.num_layers = std::stoi(input);

	std::cout << "Enter decimation ratio [0.1 - 0.9] (default: 0.63): ";
	std::getline(std::cin, input);
	if (!input.empty())
		params.decimation_ratio = std::stod(input);

	std::cout << "\n[1] Initializing Viewer and loading mesh from: " << file_path << "...\n";

	easy3d::Viewer viewer("Nested Cages Visualization");
	easy3d::Model *raw_model = viewer.add_model(file_path);
	if (!raw_model)
	{
		std::cerr << "rip: could not load mesh" << std::endl;
		return EXIT_FAILURE;
	}

	auto base_mesh = dynamic_cast<easy3d::SurfaceMesh *>(raw_model);
	if (!base_mesh)
	{
		std::cerr << "rip: not a valid surfacemesh" << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "[2] Initializing Nested Cages Algorithm...\n";
	// M0 from the paper. our starting hi-res mesh
	NestedCages::Mesh eigen_input = to_eigen(base_mesh);

	// fire up the algorithm
	NestedCages nc(eigen_input, params);

	std::cout << "[3] Computing Layers (This may take a while)...\n";

	std::vector<NestedCages::Mesh> results = nc.compute();

	std::cout << "[4] Preparing 3D Visualization...\n";
	for (size_t i = 1; i < results.size(); ++i)
	{
		SurfaceMesh *out_mesh = to_easy3d(results[i]);
		
		auto points = out_mesh->get_vertex_property<vec3>("v:point");
		for (auto v : out_mesh->vertices())
		{
			points[v] += vec3(i * 1.5f, 0.0f, 0.0f);
		}

		viewer.add_model(out_mesh);
	}

	std::cout << "Launching Viewer!\n";
	return viewer.run();
}