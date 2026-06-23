#pragma once
#include <vector>
#include <unordered_set>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#ifndef GLuint
using GLuint = unsigned int;
#endif
#ifndef GLfloat
using GLfloat = float;
#endif

class ls_mesh
{
private:
	std::vector<std::unordered_set<unsigned int>> neighbors;
	std::vector<GLuint> &orig_indices;
	std::vector<GLfloat> &orig_vertices;

	void create_neighbormap_from_indices();
	unsigned int get_most_difference_index(const std::vector<GLfloat> &vertices) const;

public:
	ls_mesh() = delete;
	ls_mesh(std::vector<GLuint> &orig_indices, std::vector<GLfloat> &orig_vertices);
	std::vector<GLfloat> calc_vertices(std::vector<unsigned int> control_points, float control_weight,
									   bool is_uniform, float *error_average) const;
	std::vector<unsigned int> random_cp(int control_count) const;
	std::vector<unsigned int> greedy_cp(int control_count, float control_weight,
										bool is_uniform) const;
	std::vector<unsigned int> iterative_greedy_cp(int control_count, float control_weight,
												  bool is_uniform) const;
};