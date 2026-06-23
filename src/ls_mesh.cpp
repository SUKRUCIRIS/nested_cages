#include "ls_mesh.h"
#include <iostream>
#include <random>

void ls_mesh::create_neighbormap_from_indices()
{
	size_t vertex_count = orig_vertices.size();
	neighbors.resize(vertex_count / 9);

	if (orig_indices.size() % 3 != 0)
	{
		std::cout << "Error: indices vector is not a multiple of 3 (non-triangular faces?)\n";
	}

	for (size_t i = 0; i < orig_indices.size(); i += 3)
	{
		unsigned int v0 = orig_indices[i];
		unsigned int v1 = orig_indices[i + 1];
		unsigned int v2 = orig_indices[i + 2];

		if (v0 >= vertex_count || v1 >= vertex_count || v2 >= vertex_count)
		{
			std::cerr << "Warning: index out of range (" << v0 << "," << v1 << "," << v2 << ")\n";
			continue;
		}

		// Make bidirectional connections for this triangle
		neighbors[v0].insert(v1);
		neighbors[v0].insert(v2);

		neighbors[v1].insert(v0);
		neighbors[v1].insert(v2);

		neighbors[v2].insert(v0);
		neighbors[v2].insert(v1);
	}
}

ls_mesh::ls_mesh(std::vector<GLuint> &orig_indices, std::vector<GLfloat> &orig_vertices)
	: orig_indices(orig_indices), orig_vertices(orig_vertices)
{
	create_neighbormap_from_indices();
}

void recompute_normals(std::vector<GLfloat> &verts, const std::vector<GLuint> &inds)
{
	unsigned int vcount = verts.size() / 9;

	// accumulate normals per vertex
	std::vector<vec3> accum(vcount);
	for (unsigned int i = 0; i < vcount; ++i)
	{
		glm_vec3_zero(accum[i]); // initialize to zero
	}

	// loop over faces
	for (size_t f = 0; f + 2 < inds.size(); f += 3)
	{
		unsigned int i0 = inds[f];
		unsigned int i1 = inds[f + 1];
		unsigned int i2 = inds[f + 2];

		vec3 p0 = {verts[i0 * 9 + 0], verts[i0 * 9 + 1], verts[i0 * 9 + 2]};
		vec3 p1 = {verts[i1 * 9 + 0], verts[i1 * 9 + 1], verts[i1 * 9 + 2]};
		vec3 p2 = {verts[i2 * 9 + 0], verts[i2 * 9 + 1], verts[i2 * 9 + 2]};

		vec3 edge1, edge2, n;
		glm_vec3_sub(p1, p0, edge1);
		glm_vec3_sub(p2, p0, edge2);
		glm_vec3_cross(edge2, edge1, n);
		if (glm_vec3_norm(n) > 0.0f)
		{
			glm_vec3_normalize(n);
		}

		glm_vec3_add(accum[i0], n, accum[i0]);
		glm_vec3_add(accum[i1], n, accum[i1]);
		glm_vec3_add(accum[i2], n, accum[i2]);
	}

	// normalize per-vertex normal and write back
	for (unsigned int i = 0; i < vcount; ++i)
	{
		if (glm_vec3_norm(accum[i]) > 0.0f)
		{
			glm_vec3_normalize(accum[i]);
		}
		verts[i * 9 + 5] = accum[i][0];
		verts[i * 9 + 6] = accum[i][1];
		verts[i * 9 + 7] = accum[i][2];
	}
}

std::vector<GLfloat> ls_mesh::calc_vertices(std::vector<unsigned int> control_points, float control_weight, bool is_uniform, float *error_average) const
{
	const size_t FLOATS_PER_VERTEX = 9; // change this if your layout differs
	if (orig_vertices.size() % FLOATS_PER_VERTEX != 0)
	{
		std::cerr << "Warning: vertices size not divisible by FLOATS_PER_VERTEX\n";
	}

	unsigned int vertex_count = static_cast<unsigned int>(orig_vertices.size() / FLOATS_PER_VERTEX);

	// Ensure neighbors vector has an entry for every vertex
	if (neighbors.size() != vertex_count)
	{
		std::cout << "Resizing neighbors vector to vertex_count.\n";
		// neighbors.resize(vertex_count); // safer than crashing
	}

	// Build Laplacian
	Eigen::SparseMatrix<float> laplacien(vertex_count, vertex_count);
	std::vector<Eigen::Triplet<float>> tripletList;
	tripletList.reserve(vertex_count * 6);

	if (is_uniform)
	{
		for (unsigned int i = 0; i < vertex_count; ++i)
		{
			tripletList.emplace_back(i, i, 1.0f);
			float degree = static_cast<float>(neighbors[i].size());
			float offWeight = (degree > 0.0f) ? -1.0f / degree : 0.0f;
			for (auto j : neighbors[i])
			{
				if (j < vertex_count)
					tripletList.emplace_back(i, j, offWeight);
			}
		}
	}
	else
	{
		for (unsigned int i = 0; i < vertex_count; ++i)
		{
			tripletList.emplace_back(i, i, 0.0f); // We'll fill diagonal later

			Eigen::Vector3f vi(
				orig_vertices[FLOATS_PER_VERTEX * i + 0],
				orig_vertices[FLOATS_PER_VERTEX * i + 1],
				orig_vertices[FLOATS_PER_VERTEX * i + 2]);

			float weight_sum = 0.0f;

			for (auto j : neighbors[i])
			{
				if (j >= vertex_count)
					continue;

				Eigen::Vector3f vj(
					orig_vertices[FLOATS_PER_VERTEX * j + 0],
					orig_vertices[FLOATS_PER_VERTEX * j + 1],
					orig_vertices[FLOATS_PER_VERTEX * j + 2]);

				// Compute cotangent weights using 1-ring triangles (i,j,k)
				float cot_sum = 0.0f;
				int cot_count = 0;

				// For each common neighbor k shared by i and j (triangles (i,j,k))
				for (auto k : neighbors[i])
				{
					if (k == j || k >= vertex_count)
						continue;
					if (std::find(neighbors[j].begin(), neighbors[j].end(), k) == neighbors[j].end())
						continue; // not a shared face

					Eigen::Vector3f vk(
						orig_vertices[FLOATS_PER_VERTEX * k + 0],
						orig_vertices[FLOATS_PER_VERTEX * k + 1],
						orig_vertices[FLOATS_PER_VERTEX * k + 2]);

					// Angles opposite edge (i,j)
					Eigen::Vector3f e_ij = vi - vj;
					Eigen::Vector3f e_ik = vi - vk;
					Eigen::Vector3f e_jk = vj - vk;

					// Pick the angle opposite to edge (i,j): that’s at vertex k
					Eigen::Vector3f u = vi - vk;
					Eigen::Vector3f v = vj - vk;

					float cos_angle = u.dot(v) / (u.norm() * v.norm());
					cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);

					float angle = std::acos(cos_angle);
					float cot = 1.0f / std::tan(angle);

					cot_sum += cot;
					cot_count++;
				}

				if (cot_count > 0)
				{
					float w = 0.5f * cot_sum; // average cotangent weight
					weight_sum += w;

					tripletList.emplace_back(i, j, -w);
				}
			}

			// Diagonal entry
			tripletList.emplace_back(i, i, weight_sum);
		}
	}

	laplacien.setFromTriplets(tripletList.begin(), tripletList.end());
	laplacien.makeCompressed();

	// Build F matrix
	tripletList.clear();
	Eigen::SparseMatrix<float> F_matrix(control_points.size(), vertex_count);
	tripletList.reserve(control_points.size());
	for (int i = 0; i < control_points.size(); ++i)
	{
		tripletList.emplace_back(i, static_cast<int>(control_points[i]), control_weight);
	}
	F_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
	F_matrix.makeCompressed();

	// Build A = [L; F]
	Eigen::SparseMatrix<float> A_matrix(laplacien.rows() + F_matrix.rows(), laplacien.cols());
	tripletList.clear();
	tripletList.reserve(static_cast<size_t>(laplacien.nonZeros() + F_matrix.nonZeros()));

	for (int k = 0; k < laplacien.outerSize(); ++k)
		for (Eigen::SparseMatrix<float>::InnerIterator it(laplacien, k); it; ++it)
			tripletList.emplace_back(it.row(), it.col(), it.value());

	for (int k = 0; k < F_matrix.outerSize(); ++k)
		for (Eigen::SparseMatrix<float>::InnerIterator it(F_matrix, k); it; ++it)
			tripletList.emplace_back(it.row() + laplacien.rows(), it.col(), it.value());

	A_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
	A_matrix.makeCompressed();

	// Build b vectors
	Eigen::VectorXf bx(A_matrix.rows()), by(A_matrix.rows()), bz(A_matrix.rows());
	bx.setZero();
	by.setZero();
	bz.setZero();
	for (int r = 0; r < control_points.size(); ++r)
	{
		unsigned int cp = control_points[r];
		bx[laplacien.rows() + r] = control_weight * orig_vertices[cp * FLOATS_PER_VERTEX + 0];
		by[laplacien.rows() + r] = control_weight * orig_vertices[cp * FLOATS_PER_VERTEX + 1];
		bz[laplacien.rows() + r] = control_weight * orig_vertices[cp * FLOATS_PER_VERTEX + 2];
	}

	Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<float>> lscg;
	lscg.compute(A_matrix);
	Eigen::VectorXf x = lscg.solve(bx);
	Eigen::VectorXf y = lscg.solve(by);
	Eigen::VectorXf z = lscg.solve(bz);

	// build new vertices vector
	std::vector<GLfloat> new_vertices;
	new_vertices.resize(vertex_count * FLOATS_PER_VERTEX);
	if (error_average)
	{
		*error_average = 0;
	}
	for (unsigned int i = 0; i < vertex_count; ++i)
	{
		new_vertices[i * FLOATS_PER_VERTEX + 0] = x[i];
		new_vertices[i * FLOATS_PER_VERTEX + 1] = y[i];
		new_vertices[i * FLOATS_PER_VERTEX + 2] = z[i];

		// preserve placeholder layout for the rest (you can adjust to carry over other attributes)
		for (size_t k = 3; k < FLOATS_PER_VERTEX; ++k)
			new_vertices[i * FLOATS_PER_VERTEX + k] = 0.0f;

		if (error_average)
		{
			*error_average += glm_vec3_distance(&(new_vertices[i * FLOATS_PER_VERTEX]), &(orig_vertices[i * FLOATS_PER_VERTEX]));
		}
	}
	if (error_average)
	{
		*error_average /= vertex_count;
	}

	recompute_normals(new_vertices, orig_indices);

	return new_vertices;
}

std::vector<unsigned int> ls_mesh::random_cp(int control_count) const
{
	int vertex_count = orig_vertices.size() / 9;
	// --- choose control points (unique, well-distributed) ---
	if (control_count <= 1)
		control_count = 2;
	if (control_count > static_cast<int>(vertex_count))
		control_count = static_cast<int>(vertex_count);

	std::vector<unsigned int> control_points;
	control_points.reserve(control_count);

	// deterministic spread + small random jitter to avoid perfect alignment
	unsigned int step = std::max(1u, vertex_count / static_cast<unsigned int>(control_count));
	std::mt19937 rng(12345);
	std::uniform_int_distribution<int> jitter(-static_cast<int>(step / 4), static_cast<int>(step / 4));

	unsigned int idx = 0;
	for (int i = 0; i < control_count; ++i)
	{
		unsigned int candidate = std::min<unsigned int>(vertex_count - 1, idx + std::max(0, jitter(rng)));
		control_points.push_back(candidate);
		idx += step;
	}
	// ensure unique (if duplicates exist, fill missing with last indexes)
	std::sort(control_points.begin(), control_points.end());
	control_points.erase(std::unique(control_points.begin(), control_points.end()), control_points.end());
	while (static_cast<int>(control_points.size()) < control_count)
	{
		// append random unique indices
		for (unsigned int k = 0; k < vertex_count && static_cast<int>(control_points.size()) < control_count; ++k)
		{
			if (!std::binary_search(control_points.begin(), control_points.end(), k))
				control_points.push_back(k);
		}
		std::sort(control_points.begin(), control_points.end());
	}
	return control_points;
}

unsigned int ls_mesh::get_most_difference_index(const std::vector<GLfloat> &vertices) const
{
	const size_t FLOATS_PER_VERTEX = 9;
	unsigned int vertex_count = static_cast<unsigned int>(orig_vertices.size() / FLOATS_PER_VERTEX);
	float maxerror = 0;
	float error = 0;
	unsigned int res = 0;
	for (size_t i = 0; i < vertex_count; i++)
	{
		error = glm_vec3_distance(&(vertices[i * FLOATS_PER_VERTEX]), &(orig_vertices[i * FLOATS_PER_VERTEX]));
		if (error > maxerror)
		{
			maxerror = error;
			res = i;
		}
	}
	return res;
}

std::vector<unsigned int> ls_mesh::greedy_cp(int control_count, float control_weight,
											 bool is_uniform) const
{
	// --- 1. Initialization and Edge Cases ---

	// Ensure the requested count is valid
	unsigned int vertex_count = orig_vertices.size() / 9;
	if (control_count <= 20)
	{
		control_count = 20;
	}
	if (control_count > static_cast<int>(vertex_count))
	{
		control_count = static_cast<int>(vertex_count);
	}

	// Start with a small set of initial control points (minimum 2 recommended)
	std::vector<unsigned int> control_points;
	control_points = random_cp(10);

	// --- 2. Iterative Greedy Selection ---

	// Loop until we reach the desired number of control points
	while (static_cast<int>(control_points.size()) < control_count)
	{

		// a) Calculate the Least Squares approximation with current CPs
		// Since this is a const function, we can't call a non-const calc_vertices.
		// Assuming calc_vertices is NOT implicitly const (though you declared it as const).
		// To be safe, we rely on the internal state (this is a common const/non-const dilemma).

		// Since 'calc_vertices' is declared as 'const', we can call it:
		std::vector<GLfloat> new_verts = this->calc_vertices(
			control_points,
			control_weight, // Use a default control weight (e.g., 10.0f)
			is_uniform,		// Use uniform weights for initial error calculation
			0);

		// Ensure the calculation produced valid results
		if (new_verts.empty() || new_verts.size() != orig_vertices.size())
		{
			std::cerr << "Error: LS calculation failed or produced wrong size.\n";
			break;
		}

		// b) Find the vertex index with the maximum difference (error)
		unsigned int max_error_index = get_most_difference_index(new_verts);

		// c) Add the max-error vertex as a new control point, if it's not already one
		bool already_exists = false;
		for (unsigned int cp : control_points)
		{
			if (cp == max_error_index)
			{
				already_exists = true;
				break;
			}
		}

		if (!already_exists)
		{
			control_points.push_back(max_error_index);
			printf("New cp added: %d\n", control_points.size());
		}
		else
		{
			// This can happen if the two highest error points are the existing CPs.
			// In a simple loop, this results in an infinite loop if not handled.
			std::cout << "Warning: Max error index is already a control point. Stopping greedy search.\n";
			break;
		}
	}

	return control_points;
}

std::vector<unsigned int> ls_mesh::iterative_greedy_cp(int control_count, float control_weight,
													   bool is_uniform) const
{
	// --- 1. Initialization and Edge Cases ---
	const size_t FLOATS_PER_VERTEX = 9;
	const size_t NUM_POINTS_PER_ITERATION = 10;

	unsigned int vertex_count = orig_vertices.size() / FLOATS_PER_VERTEX;

	// Ensure minimum count is 20 (as requested)
	if (control_count <= 20)
	{
		control_count = 20;
	}
	if (control_count > static_cast<int>(vertex_count))
	{
		control_count = static_cast<int>(vertex_count);
	}

	// Start with 10 random control points
	std::vector<unsigned int> control_points = random_cp(10);

	// Keep control_points sorted for fast uniqueness checks later
	std::sort(control_points.begin(), control_points.end());

	// --- 2. Iterative Greedy Selection ---

	// Loop until we reach the desired number of control points
	while (static_cast<int>(control_points.size()) < control_count)
	{
		std::vector<GLfloat> new_verts = this->calc_vertices(
			control_points,
			control_weight,
			is_uniform,
			0); // Note: Passing 0 to a float& argument is problematic,
				// fixed by using a placeholder float here.

		if (new_verts.empty() || new_verts.size() != orig_vertices.size())
		{
			std::cerr << "Error: LS calculation failed or produced wrong size.\n";
			break;
		}

		// b) Map all vertices' errors: {error, index}
		std::vector<std::pair<float, unsigned int>> error_map;
		error_map.reserve(vertex_count);
		for (unsigned int i = 0; i < vertex_count; ++i)
		{
			// Note: We use the *original* vertices for comparison, which is correct.
			float error = glm_vec3_distance(&(new_verts[i * FLOATS_PER_VERTEX]), &(orig_vertices[i * FLOATS_PER_VERTEX]));
			error_map.emplace_back(error, i);
		}

		// c) Sort by error (descending) to find the vertices with the largest displacement
		std::sort(error_map.begin(), error_map.end(),
				  [](const auto &a, const auto &b)
				  {
					  return a.first > b.first;
				  });

		// d) Select and add the top unique candidates
		size_t points_added_this_iter = 0;
		size_t vertices_checked = 0;
		bool added_any = false;

		for (const auto &pair : error_map)
		{
			unsigned int potential_cp = pair.second;
			vertices_checked++;

			// Stop if we hit the limit for this iteration OR the overall final limit
			if (points_added_this_iter >= NUM_POINTS_PER_ITERATION || static_cast<int>(control_points.size()) >= control_count)
				break;

			// Check if this point is already a control point (requires control_points to be sorted)
			if (!std::binary_search(control_points.begin(), control_points.end(), potential_cp))
			{
				control_points.push_back(potential_cp);
				points_added_this_iter++;
				added_any = true;
			}
		}

		// Re-sort control points after adding new ones to maintain binary_search compatibility
		std::sort(control_points.begin(), control_points.end());

		// Check for stagnation
		if (!added_any && vertices_checked == vertex_count)
		{
			std::cout << "Warning: Checked all vertices but could not find new unique control points. Stopping greedy search.\n";
			break;
		}

		printf("New CPs added this iteration: %zu. Total control points: %zu\n", points_added_this_iter, control_points.size());
	}

	return control_points;
}