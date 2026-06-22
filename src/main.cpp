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

#include <limits>
#include <map>

// Software-level clipping to extract half of the mesh dynamically
easy3d::SurfaceMesh *create_half_mesh(easy3d::SurfaceMesh *src)
{
	auto half = new easy3d::SurfaceMesh();
	auto src_points = src->get_vertex_property<vec3>("v:point");

	// 1. Find the actual X-axis bounding box of the specific mesh
	float min_x = std::numeric_limits<float>::max();
	float max_x = std::numeric_limits<float>::lowest();

	for (auto v : src->vertices())
	{
		float x = src_points[v].x;
		if (x < min_x)
			min_x = x;
		if (x > max_x)
			max_x = x;
	}

	// Calculate the absolute center of the model
	float center_x = (min_x + max_x) / 2.0f;

	// Track which original vertices have been added to avoid duplicates
	std::map<easy3d::SurfaceMesh::Vertex, easy3d::SurfaceMesh::Vertex> vmap;

	for (auto f : src->faces())
	{
		vec3 centroid(0, 0, 0);
		int vcount = 0;
		for (auto v : src->vertices(f))
		{
			centroid += src_points[v];
			vcount++;
		}
		centroid /= static_cast<float>(vcount);

		// 2. Keep faces on the right side of the bounding center
		if (centroid.x > center_x)
		{
			std::vector<easy3d::SurfaceMesh::Vertex> face_v;
			for (auto v : src->vertices(f))
			{
				if (vmap.find(v) == vmap.end())
				{
					vmap[v] = half->add_vertex(src_points[v]);
				}
				face_v.push_back(vmap[v]);
			}
			half->add_face(face_v);
		}
	}
	return half;
}

#include <easy3d/renderer/renderer.h>

class CageViewer : public easy3d::Viewer
{
	easy3d::Model *layer1_full = nullptr;
	easy3d::Model *layer1_half = nullptr;
	bool showing_half = false;

public:
	CageViewer(const std::string &title) : easy3d::Viewer(title) {}

	void set_layer1_models(easy3d::Model *full, easy3d::Model *half)
	{
		layer1_full = full;
		layer1_half = half;

		// Hide the half mesh by default by talking to its renderer
		if (layer1_half && layer1_half->renderer())
		{
			layer1_half->renderer()->set_visible(false);
		}
	}

	bool key_press_event(int key, int modifiers) override
	{
		// Intercept 'H' or 'h' key
		if (key == 'H' || key == 'h')
		{
			if (layer1_full && layer1_half)
			{
				showing_half = !showing_half;

				if (layer1_full->renderer())
				{
					layer1_full->renderer()->set_visible(!showing_half);
				}
				if (layer1_half->renderer())
				{
					layer1_half->renderer()->set_visible(showing_half);
				}

				std::cout << (showing_half ? "[Toggled] Showing Half Layer 1\n" : "[Toggled] Showing Full Layer 1\n");
				update(); // Force viewer to redraw
			}
			return true;
		}
		// Pass any other key presses back to the default handler
		return easy3d::Viewer::key_press_event(key, modifiers);
	}
};

#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/renderer/state.h>

void set_model_color(easy3d::Model *model, const easy3d::vec4 &color)
{
	if (!model || !model->renderer())
		return;

	auto drawable = model->renderer()->get_triangles_drawable("faces");
	if (drawable)
	{
		drawable->set_coloring_method(easy3d::State::UNIFORM_COLOR);
		drawable->set_color(color);
	}
}

int main(int argc, char **argv)
{
	std::cout << "========================================\n";
	std::cout << "      NESTED CAGES GENERATOR            \n";
	std::cout << "========================================\n\n";

	std::cout << "Select execution mode:\n";
	std::cout << "[1] Calculate new nested cages\n";
	std::cout << "[2] Load existing layers (homer.obj & layer_1.obj)\n";
	std::cout << "Choice (default: 1): ";

	std::string input;
	std::getline(std::cin, input);
	int mode = input.empty() ? 1 : std::stoi(input);

	CageViewer viewer("Nested Cages Visualization");

	easy3d::Model *l1_full_model = nullptr;
	easy3d::Model *l1_half_model = nullptr;

	if (mode == 1)
	{
		std::string file_path = "homer.obj";
		NestedCages::Parameters params;
		params.num_layers = 1;
		params.decimation_ratio = 0.63;
		params.flow_dt = 0.001;

		std::cout << "Enter mesh file path (default: homer.obj): ";
		std::getline(std::cin, input);
		if (!input.empty())
			file_path = input;

		std::cout << "\n[1] Initializing Viewer and loading mesh...\n";
		easy3d::Model *raw_model = viewer.add_model(file_path);

		// Color the base mesh RED
		set_model_color(raw_model, easy3d::vec4(1.0f, 0.0f, 0.0f, 1.0f));

		auto base_mesh = dynamic_cast<easy3d::SurfaceMesh *>(raw_model);
		if (!base_mesh)
		{
			std::cerr << "rip: could not load mesh" << std::endl;
			return EXIT_FAILURE;
		}

		std::cout << "[2] Initializing Nested Cages Algorithm...\n";
		NestedCages::Mesh eigen_input = to_eigen(base_mesh);
		NestedCages nc(eigen_input, params);

		std::cout << "[3] Computing Layers (This may take a while)...\n";
		std::vector<NestedCages::Mesh> results = nc.compute();

		std::cout << "[4] Preparing 3D Visualization...\n";
		for (size_t i = 1; i < results.size(); ++i)
		{
			SurfaceMesh *out_mesh = to_easy3d(results[i]);
			auto added_model = viewer.add_model(out_mesh);

			// If this is Layer 1, build the half-mesh and color both YELLOW
			if (i == 1)
			{
				set_model_color(added_model, easy3d::vec4(1.0f, 1.0f, 0.0f, 1.0f));
				l1_full_model = added_model;

				SurfaceMesh *half_mesh = create_half_mesh(out_mesh);
				l1_half_model = viewer.add_model(half_mesh);
				set_model_color(l1_half_model, easy3d::vec4(1.0f, 1.0f, 0.0f, 1.0f));
			}
		}
	}
	else if (mode == 2)
	{
		std::cout << "\nLoading existing files...\n";

		easy3d::Model *base_model = viewer.add_model("homer.obj");
		if (base_model)
		{
			std::cout << "Loaded homer.obj\n";
			set_model_color(base_model, easy3d::vec4(1.0f, 0.0f, 0.0f, 1.0f)); // RED
		}

		easy3d::Model *layer_model = viewer.add_model("layer_1.obj");
		if (layer_model)
		{
			std::cout << "Loaded layer_1.obj\n";
			set_model_color(layer_model, easy3d::vec4(1.0f, 1.0f, 0.0f, 1.0f)); // YELLOW
			l1_full_model = layer_model;

			auto sm = dynamic_cast<easy3d::SurfaceMesh *>(layer_model);
			if (sm)
			{
				SurfaceMesh *half_mesh = create_half_mesh(sm);
				l1_half_model = viewer.add_model(half_mesh);
				set_model_color(l1_half_model, easy3d::vec4(1.0f, 1.0f, 0.0f, 1.0f)); // YELLOW
			}
		}
	}

	viewer.set_layer1_models(l1_full_model, l1_half_model);

	std::cout << "\nControls:\n";
	std::cout << " - Press 'H' to toggle half-visibility for Layer 1.\n\n";
	std::cout << "Launching Viewer!\n";

	return viewer.run();
}