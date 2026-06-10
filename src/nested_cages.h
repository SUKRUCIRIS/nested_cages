#pragma once

#include <vector>
#include <Eigen/Core>
#include <igl/arap.h>
#include <ipc/ipc.hpp>
#include <ipc/collision_mesh.hpp>

class NestedCages
{
public:
	struct Mesh
	{
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
	};

	struct Parameters
	{
		int num_layers = 1;
		double decimation_ratio = 0.63;
		double flow_dt = 0.05;
		int max_flow_iters = 1000;
		double reinflation_beta_init = 0.01;
		double ipc_dhat = 0.001;
	};

	NestedCages(const Mesh &input_mesh, const Parameters &params);
	std::vector<Mesh> compute();

private:
	Mesh m_input_mesh;
	Parameters m_params;

	std::vector<Eigen::MatrixXd> shrink(const Mesh &C_hat, const Mesh &F);
	Mesh reinflate(std::vector<Eigen::MatrixXd> &H, const Mesh &C_hat, const Mesh &F_original);
	Mesh decimate(const Mesh &input, double target_ratio);

	void build_ipc_mesh(const Eigen::MatrixXd &C_V, const Eigen::MatrixXi &C_F,
						const Eigen::MatrixXd &F_V, const Eigen::MatrixXi &F_F,
						Eigen::MatrixXd &combined_V, Eigen::MatrixXi &combined_E, Eigen::MatrixXi &combined_F);
};