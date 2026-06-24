#include <easy3d/viewer/viewer.h>
#include <easy3d/core/surface_mesh.h>
#include <easy3d/core/types.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/state.h>

#include "nested_cages.h"
#include "ls_mesh.h"
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <map>
#include <chrono>
#include <algorithm>

#ifndef GLuint
using GLuint = unsigned int;
#endif
#ifndef GLfloat
using GLfloat = float;
#endif

using namespace easy3d;

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

easy3d::SurfaceMesh *create_half_mesh(easy3d::SurfaceMesh *src)
{
	auto half = new easy3d::SurfaceMesh();
	auto src_points = src->get_vertex_property<vec3>("v:point");

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

	float center_x = (min_x + max_x) / 2.0f;
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

class CustomViewer : public easy3d::Viewer
{
public:
	easy3d::Model *l1_full = nullptr;
	easy3d::Model *l1_half = nullptr;

	std::vector<easy3d::Model *> cp_models;

	bool is_ls_mode = false;
	bool showing_half = false;
	bool showing_cp = true;

	CustomViewer(const std::string &title) : easy3d::Viewer(title) {}

	bool key_press_event(int key, int modifiers) override
	{
		if (is_ls_mode)
		{
			if (key == 'C' || key == 'c')
			{
				showing_cp = !showing_cp;
				for (auto cp_model : cp_models)
				{
					if (cp_model && cp_model->renderer())
					{
						auto pt_draw = cp_model->renderer()->get_points_drawable("vertices");
						if (pt_draw)
							pt_draw->set_visible(showing_cp);
					}
				}
				std::cout << (showing_cp ? "[Toggled] Showing Control Points\n" : "[Toggled] Hiding Control Points\n");
				update();
				return true;
			}
		}
		else
		{
			if (key == 'H' || key == 'h')
			{
				if (l1_full && l1_half)
				{
					showing_half = !showing_half;

					if (l1_full->renderer())
						l1_full->renderer()->set_visible(!showing_half);
					if (l1_half->renderer())
						l1_half->renderer()->set_visible(showing_half);

					std::cout << (showing_half ? "[Toggled] Showing Half Layer 1\n" : "[Toggled] Showing Full Layer 1\n");
					update();
				}
				return true;
			}
		}
		return easy3d::Viewer::key_press_event(key, modifiers);
	}
};

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
	std::cout << "  NESTED CAGES & LS MESH OPTIMIZATION   \n";
	std::cout << "========================================\n\n";

	std::cout << "Select execution mode:\n";
	std::cout << "[1] Calculate new nested cages\n";
	std::cout << "[2] Load existing layers\n";
	std::cout << "Choice (default: 1): ";

	std::string input;
	std::getline(std::cin, input);
	int mode = input.empty() ? 1 : std::stoi(input);

	// Original File Prompt (Applies to both modes now)
	std::string file_path = "homer.obj";
	std::cout << "Enter original mesh file path (default: homer.obj): ";
	std::getline(std::cin, input);
	if (!input.empty())
		file_path = input;

	float decimation_ratio = 0.63f;
	if (mode == 1)
	{
		std::cout << "Enter decimation percentage for Layer 1 cage (e.g. 63 for 63%, default: 63): ";
		std::getline(std::cin, input);
		if (!input.empty())
			decimation_ratio = std::stof(input) / 100.0f;
	}

	std::cout << "\nDo you want to continue with Least Squares Mesh Optimization? [1=Yes, 0=No] (default: 1): ";
	std::getline(std::cin, input);
	int run_ls = input.empty() ? 1 : std::stoi(input);

	int cp_type = 1;
	int cp_count = 20;

	if (run_ls == 1)
	{
		std::cout << "Enter Control Point count (default: 20): ";
		std::getline(std::cin, input);
		if (!input.empty())
			cp_count = std::stoi(input);

		std::cout << "Select Control Point Algorithm:\n";
		std::cout << "1: Greedy (default)\n2: Random\n3: Iterative Greedy\nChoice: ";
		std::getline(std::cin, input);
		if (!input.empty())
			cp_type = std::stoi(input);
	}

	CustomViewer viewer("Mesh Viewer");

	easy3d::Model *raw_model = nullptr;
	easy3d::Model *l1_full_model = nullptr;
	easy3d::Model *l1_half_model = nullptr;

	if (mode == 1)
	{
		NestedCages::Parameters params;
		params.num_layers = 1;
		params.decimation_ratio = decimation_ratio;
		params.flow_dt = 0.001;

		std::cout << "\n[1] Initializing Viewer and loading mesh...\n";
		raw_model = viewer.add_model(file_path);
		set_model_color(raw_model, easy3d::vec4(1.0f, 0.0f, 0.0f, 1.0f));

		auto base_mesh = dynamic_cast<easy3d::SurfaceMesh *>(raw_model);
		if (!base_mesh)
			return EXIT_FAILURE;

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

			if (i == 1)
			{
				set_model_color(added_model, easy3d::vec4(1.0f, 1.0f, 0.0f, 1.0f));
				l1_full_model = added_model;

				if (run_ls == 0)
				{
					SurfaceMesh *half_mesh = create_half_mesh(out_mesh);
					l1_half_model = viewer.add_model(half_mesh);
					set_model_color(l1_half_model, easy3d::vec4(1.0f, 1.0f, 0.0f, 1.0f));
				}
			}
		}
	}
	else if (mode == 2)
	{
		std::cout << "\nLoading existing files...\n";

		// Use the dynamically provided file_path
		raw_model = viewer.add_model(file_path);
		if (raw_model)
			set_model_color(raw_model, easy3d::vec4(1.0f, 0.0f, 0.0f, 1.0f));

		// Layer 1 is guaranteed to be layer_1.obj
		easy3d::Model *layer_model = viewer.add_model("layer_1.obj");
		if (layer_model)
		{
			std::cout << "Loaded layer_1.obj\n";
			set_model_color(layer_model, easy3d::vec4(1.0f, 1.0f, 0.0f, 1.0f));
			l1_full_model = layer_model;

			if (run_ls == 0)
			{
				auto sm = dynamic_cast<easy3d::SurfaceMesh *>(layer_model);
				if (sm)
				{
					SurfaceMesh *half_mesh = create_half_mesh(sm);
					l1_half_model = viewer.add_model(half_mesh);
					set_model_color(l1_half_model, easy3d::vec4(1.0f, 1.0f, 0.0f, 1.0f));
				}
			}
		}
	}

	// --- LEAST SQUARES MESH OPTIMIZATION COMPARISON ---
	if (run_ls == 1 && raw_model && l1_full_model)
	{
		viewer.is_ls_mode = true;

		std::cout << "\n========================================\n";
		std::cout << "  RUNNING LS MESH PIPELINE COMPARISON   \n";
		std::cout << "========================================\n";

		// Hide the source models; we will build our own shifted display meshes
		if (raw_model->renderer())
			raw_model->renderer()->set_visible(false);
		if (l1_full_model->renderer())
			l1_full_model->renderer()->set_visible(false);
		if (l1_half_model && l1_half_model->renderer())
			l1_half_model->renderer()->set_visible(false);

		auto orig_mesh = dynamic_cast<easy3d::SurfaceMesh *>(raw_model);
		auto l1_mesh = dynamic_cast<easy3d::SurfaceMesh *>(l1_full_model);

		// Extract Original Mesh Data & Calculate Bounding Limits
		std::vector<GLuint> orig_indices;
		std::vector<GLfloat> orig_vertices;
		float min_x = std::numeric_limits<float>::max();
		float max_x = std::numeric_limits<float>::lowest();
		float min_z = std::numeric_limits<float>::max();
		float max_z = std::numeric_limits<float>::lowest();

		auto orig_points = orig_mesh->get_vertex_property<vec3>("v:point");
		for (auto v : orig_mesh->vertices())
		{
			float x = orig_points[v].x;
			float y = orig_points[v].y;
			float z = orig_points[v].z;

			if (x < min_x)
				min_x = x;
			if (x > max_x)
				max_x = x;
			if (z < min_z)
				min_z = z;
			if (z > max_z)
				max_z = z;

			orig_vertices.push_back(x);
			orig_vertices.push_back(y);
			orig_vertices.push_back(z);
			for (int i = 0; i < 6; i++)
				orig_vertices.push_back(0.0f);
		}
		for (auto f : orig_mesh->faces())
		{
			int c = 0;
			for (auto v : orig_mesh->vertices(f))
			{
				if (c < 3)
					orig_indices.push_back(v.idx());
				c++;
			}
		}

		// Extract Layer 1 Data
		std::vector<GLuint> l1_indices;
		std::vector<GLfloat> l1_vertices;
		auto l1_points = l1_mesh->get_vertex_property<vec3>("v:point");
		for (auto v : l1_mesh->vertices())
		{
			l1_vertices.push_back(l1_points[v].x);
			l1_vertices.push_back(l1_points[v].y);
			l1_vertices.push_back(l1_points[v].z);
			for (int i = 0; i < 6; i++)
				l1_vertices.push_back(0.0f);
		}
		for (auto f : l1_mesh->faces())
		{
			int c = 0;
			for (auto v : l1_mesh->vertices(f))
			{
				if (c < 3)
					l1_indices.push_back(v.idx());
				c++;
			}
		}

		// Calculate offsets for Side-by-Side (X) and Background (Z) viewing
		float x_span = (max_x - min_x) * 1.25f;
		float offset_1 = x_span;		// Middle (Regular LS)
		float offset_2 = x_span * 2.0f; // Right (Your Method LS)

		float z_span = (max_z - min_z) * 1.5f;
		float z_offset_behind = -z_span; // Push backwards along Z axis

		// ==========================================
		// PIPELINE A: REGULAR LS (Original Mesh Only)
		// ==========================================
		std::cout << "\n[PIPELINE A] Regular LS Computation...\n";
		auto tA_start = std::chrono::high_resolution_clock::now();

		ls_mesh orig_ls(orig_indices, orig_vertices);
		std::vector<unsigned int> reg_cps;
		if (cp_type == 1)
			reg_cps = orig_ls.greedy_cp(cp_count, 10.0f, true);
		else if (cp_type == 2)
			reg_cps = orig_ls.random_cp(cp_count);
		else
			reg_cps = orig_ls.iterative_greedy_cp(cp_count, 10.0f, true);

		float reg_error = 0.0f;
		std::vector<GLfloat> reg_new_vertices = orig_ls.calc_vertices(reg_cps, 10.0f, true, &reg_error);

		auto tA_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> tA_total = tA_end - tA_start;
		std::cout << "  -> Total Time: " << tA_total.count() << " seconds\n";
		std::cout << "  -> Reconstruction Error: " << reg_error << "\n";

		// ==========================================
		// PIPELINE B: YOUR METHOD (Proxy via Layer 1)
		// ==========================================
		std::cout << "\n[PIPELINE B] 'My Method' Computation...\n";
		auto tB_start = std::chrono::high_resolution_clock::now();

		// Step 1: Find Control Points on Layer 1 Proxy
		ls_mesh l1_ls(l1_indices, l1_vertices);
		std::vector<unsigned int> l1_cps;
		if (cp_type == 1)
			l1_cps = l1_ls.greedy_cp(cp_count, 10.0f, true);
		else if (cp_type == 2)
			l1_cps = l1_ls.random_cp(cp_count);
		else
			l1_cps = l1_ls.iterative_greedy_cp(cp_count, 10.0f, true);

		auto tB_s1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> tB_step1 = tB_s1 - tB_start;
		std::cout << "  -> Step 1 (Find CP on L1): " << tB_step1.count() << " sec\n";

		// Step 2: Map Layer 1 Control Points to nearest Original Mesh vertices
		std::vector<unsigned int> mapped_cps;
		unsigned int orig_v_count = orig_vertices.size() / 9;

		for (unsigned int l1_cp : l1_cps)
		{
			float l1_x = l1_vertices[l1_cp * 9 + 0];
			float l1_y = l1_vertices[l1_cp * 9 + 1];
			float l1_z = l1_vertices[l1_cp * 9 + 2];

			float min_dist2 = std::numeric_limits<float>::max();
			unsigned int best_orig_idx = 0;

			for (unsigned int i = 0; i < orig_v_count; ++i)
			{
				float dx = orig_vertices[i * 9 + 0] - l1_x;
				float dy = orig_vertices[i * 9 + 1] - l1_y;
				float dz = orig_vertices[i * 9 + 2] - l1_z;
				float dist2 = dx * dx + dy * dy + dz * dz;
				if (dist2 < min_dist2)
				{
					min_dist2 = dist2;
					best_orig_idx = i;
				}
			}
			mapped_cps.push_back(best_orig_idx);
		}

		// Ensure uniqueness if multiple proxy points snap to the same high-res vertex
		std::sort(mapped_cps.begin(), mapped_cps.end());
		mapped_cps.erase(std::unique(mapped_cps.begin(), mapped_cps.end()), mapped_cps.end());

		auto tB_s2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> tB_step2 = tB_s2 - tB_s1;
		std::cout << "  -> Step 2 (Nearest-Neighbor Mapping): " << tB_step2.count() << " sec (" << mapped_cps.size() << " unique CPs)\n";

		// Step 3: Run Full LS on Original using Mapped CPs
		float my_error = 0.0f;
		std::vector<GLfloat> my_new_vertices = orig_ls.calc_vertices(mapped_cps, 10.0f, true, &my_error);

		auto tB_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> tB_step3 = tB_end - tB_s2;
		std::chrono::duration<double> tB_total = tB_end - tB_start;
		std::cout << "  -> Step 3 (LS Solve on Original): " << tB_step3.count() << " sec\n";
		std::cout << "  -> Total Time: " << tB_total.count() << " seconds\n";
		std::cout << "  -> Reconstruction Error: " << my_error << "\n";

		// ==========================================
		// VISUALIZATION GENERATION
		// ==========================================

		// Display 1: Original Mesh (Left, X-Offset = 0, Red)
		easy3d::SurfaceMesh *d1_emesh = new easy3d::SurfaceMesh();
		std::vector<easy3d::SurfaceMesh::Vertex> d1_vh;
		for (size_t i = 0; i < orig_vertices.size() / 9; ++i)
		{
			d1_vh.push_back(d1_emesh->add_vertex(vec3(orig_vertices[i * 9 + 0], orig_vertices[i * 9 + 1], orig_vertices[i * 9 + 2])));
		}
		for (size_t i = 0; i < orig_indices.size(); i += 3)
		{
			d1_emesh->add_face({d1_vh[orig_indices[i]], d1_vh[orig_indices[i + 1]], d1_vh[orig_indices[i + 2]]});
		}
		auto d1_model = viewer.add_model(d1_emesh);
		set_model_color(d1_model, easy3d::vec4(1.0f, 0.0f, 0.0f, 1.0f));

		// Display 2: Regular LS Mesh (Center, X-Offset = offset_1, Blue)
		easy3d::SurfaceMesh *d2_emesh = new easy3d::SurfaceMesh();
		std::vector<easy3d::SurfaceMesh::Vertex> d2_vh;
		for (size_t i = 0; i < reg_new_vertices.size() / 9; ++i)
		{
			d2_vh.push_back(d2_emesh->add_vertex(vec3(reg_new_vertices[i * 9 + 0] + offset_1, reg_new_vertices[i * 9 + 1], reg_new_vertices[i * 9 + 2])));
		}
		for (size_t i = 0; i < orig_indices.size(); i += 3)
		{
			d2_emesh->add_face({d2_vh[orig_indices[i]], d2_vh[orig_indices[i + 1]], d2_vh[orig_indices[i + 2]]});
		}
		auto d2_model = viewer.add_model(d2_emesh);
		set_model_color(d2_model, easy3d::vec4(0.2f, 0.6f, 1.0f, 1.0f));

		easy3d::SurfaceMesh *d2_cp_emesh = new easy3d::SurfaceMesh();
		for (auto cp : reg_cps)
		{
			d2_cp_emesh->add_vertex(vec3(orig_vertices[cp * 9 + 0] + offset_1, orig_vertices[cp * 9 + 1], orig_vertices[cp * 9 + 2]));
		}
		auto d2_cp_model = viewer.add_model(d2_cp_emesh);
		viewer.cp_models.push_back(d2_cp_model);

		// Display 3: Your Method LS Mesh (Right, X-Offset = offset_2, Cyan)
		easy3d::SurfaceMesh *d3_emesh = new easy3d::SurfaceMesh();
		std::vector<easy3d::SurfaceMesh::Vertex> d3_vh;
		for (size_t i = 0; i < my_new_vertices.size() / 9; ++i)
		{
			d3_vh.push_back(d3_emesh->add_vertex(vec3(my_new_vertices[i * 9 + 0] + offset_2, my_new_vertices[i * 9 + 1], my_new_vertices[i * 9 + 2])));
		}
		for (size_t i = 0; i < orig_indices.size(); i += 3)
		{
			d3_emesh->add_face({d3_vh[orig_indices[i]], d3_vh[orig_indices[i + 1]], d3_vh[orig_indices[i + 2]]});
		}
		auto d3_model = viewer.add_model(d3_emesh);
		set_model_color(d3_model, easy3d::vec4(0.0f, 0.8f, 0.8f, 1.0f));

		easy3d::SurfaceMesh *d3_cp_emesh = new easy3d::SurfaceMesh();
		for (auto cp : mapped_cps)
		{
			d3_cp_emesh->add_vertex(vec3(orig_vertices[cp * 9 + 0] + offset_2, orig_vertices[cp * 9 + 1], orig_vertices[cp * 9 + 2]));
		}
		auto d3_cp_model = viewer.add_model(d3_cp_emesh);
		viewer.cp_models.push_back(d3_cp_model);

		// Display 4: Layer 1 Proxy Cage (Behind Display 3, Z-Offset = z_offset_behind, Yellow)
		easy3d::SurfaceMesh *d4_emesh = new easy3d::SurfaceMesh();
		std::vector<easy3d::SurfaceMesh::Vertex> d4_vh;
		for (size_t i = 0; i < l1_vertices.size() / 9; ++i)
		{
			d4_vh.push_back(d4_emesh->add_vertex(vec3(
				l1_vertices[i * 9 + 0] + offset_2,
				l1_vertices[i * 9 + 1],
				l1_vertices[i * 9 + 2] + z_offset_behind)));
		}
		for (size_t i = 0; i < l1_indices.size(); i += 3)
		{
			d4_emesh->add_face({d4_vh[l1_indices[i]], d4_vh[l1_indices[i + 1]], d4_vh[l1_indices[i + 2]]});
		}
		auto d4_model = viewer.add_model(d4_emesh);
		set_model_color(d4_model, easy3d::vec4(1.0f, 1.0f, 0.0f, 1.0f));

		easy3d::SurfaceMesh *d4_cp_emesh = new easy3d::SurfaceMesh();
		for (auto cp : l1_cps)
		{
			d4_cp_emesh->add_vertex(vec3(
				l1_vertices[cp * 9 + 0] + offset_2,
				l1_vertices[cp * 9 + 1],
				l1_vertices[cp * 9 + 2] + z_offset_behind));
		}
		auto d4_cp_model = viewer.add_model(d4_cp_emesh);
		viewer.cp_models.push_back(d4_cp_model);

		// Global CP Renderer Setup
		for (auto cp_mod : viewer.cp_models)
		{
			if (cp_mod && cp_mod->renderer())
			{
				auto pt_draw = cp_mod->renderer()->get_points_drawable("vertices");
				if (pt_draw)
				{
					pt_draw->set_visible(true);
					pt_draw->set_point_size(10.0f);
					pt_draw->set_coloring_method(easy3d::State::UNIFORM_COLOR);
					pt_draw->set_color(easy3d::vec4(0.0f, 1.0f, 0.0f, 1.0f)); // Green dots
				}
				auto face_draw = cp_mod->renderer()->get_triangles_drawable("faces");
				if (face_draw)
					face_draw->set_visible(false);
				auto edge_draw = cp_mod->renderer()->get_lines_drawable("edges");
				if (edge_draw)
					edge_draw->set_visible(false);
			}
		}
	}
	else if (run_ls == 0)
	{
		viewer.is_ls_mode = false;
		viewer.l1_full = l1_full_model;
		viewer.l1_half = l1_half_model;

		if (l1_half_model && l1_half_model->renderer())
			l1_half_model->renderer()->set_visible(false);
	}

	std::cout << "\nControls:\n";
	if (viewer.is_ls_mode)
	{
		std::cout << " - Press 'C' to toggle Control Points visibility.\n";
	}
	else
	{
		std::cout << " - Press 'H' to toggle half-visibility for Layer 1.\n";
	}
	std::cout << "\nLaunching Viewer!\n";

	return viewer.run();
}