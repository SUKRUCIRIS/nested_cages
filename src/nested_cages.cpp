#include <cassert>
#include <cmath>
#include <algorithm>
#include <string>

#include "nested_cages.h"
#include <igl/decimate.h>
#include <igl/AABB.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>
#include <igl/edges.h>
#include <igl/facet_components.h>
#include <igl/doublearea.h>
#include <igl/writeOBJ.h>
#include <iostream>

#include <ipc/ipc.hpp>
#include <ipc/collision_mesh.hpp>

NestedCages::NestedCages(const Mesh &input_mesh, const Parameters &params)
	: m_input_mesh(input_mesh), m_params(params) {}

std::vector<NestedCages::Mesh> NestedCages::compute()
{
	std::vector<Mesh> layers;
	layers.push_back(m_input_mesh);

	for (int i = 1; i <= m_params.num_layers; ++i)
	{
		std::cout << "\n========================================\n";
		std::cout << " Computing Layer " << i << " / " << m_params.num_layers << "\n";
		std::cout << "========================================\n";

		Mesh M_prev = layers.back();

		std::cout << "  [1/3] Decimating coarse cage...\n";

		Mesh C_hat = decimate(M_prev, m_params.decimation_ratio);
		Mesh F = M_prev;

		std::cout << "  [2/3] Flowing fine mesh inside coarse cage...\n";

		std::vector<Eigen::MatrixXd> H = shrink(C_hat, F);

		std::cout << "  [3/3] Re-inflating with ARAP & IPC CCD (Frames: " << H.size() << ")...\n";

		Mesh M_i = reinflate(H, C_hat, F);

		layers.push_back(M_i);

		std::string filename = "layer_" + std::to_string(i) + ".obj";
		igl::writeOBJ(filename, M_i.V, M_i.F);
		std::cout << "  [SAVED] Layer " << i << " successfully written to " << filename << "\n";
	}
	return layers;
}

NestedCages::Mesh NestedCages::decimate(const Mesh &input, double target_ratio)
{
	Mesh output;
	Eigen::VectorXi J, I;

	int target_faces = std::max(4, static_cast<int>(input.F.rows() * target_ratio));

	igl::decimate(input.V, input.F, target_faces, false, output.V, output.F, J, I);
	return output;
}

std::vector<Eigen::MatrixXd> NestedCages::shrink(const Mesh &C_hat, const Mesh &F)
{
	std::vector<Eigen::MatrixXd> H;
	Eigen::MatrixXd F_curr = F.V;
	H.push_back(F_curr);

	igl::AABB<Eigen::MatrixXd, 3> tree;
	tree.init(C_hat.V, C_hat.F);

	Eigen::MatrixXd FN;
	igl::per_face_normals(C_hat.V, C_hat.F, FN);

	for (int i = 0; i < FN.rows(); ++i)
	{
		if (!FN.row(i).allFinite())
			FN.row(i) = Eigen::RowVector3d(1, 0, 0);
	}

	std::vector<Eigen::Vector3d> quad_bary = {
		Eigen::Vector3d(0.5, 0.5, 0.0),
		Eigen::Vector3d(0.0, 0.5, 0.5),
		Eigen::Vector3d(0.5, 0.0, 0.5)};
	double quad_weight = 1.0 / 3.0;

	Eigen::MatrixXd grad_Phi(F_curr.rows(), 3);

	Eigen::MatrixXd p_quad(1, 3);
	Eigen::MatrixXd C_closest(1, 3);
	Eigen::VectorXi I_face(1);
	Eigen::VectorXd sqrD(1);

	for (int iter = 0; iter < m_params.max_flow_iters; ++iter)
	{
		grad_Phi.setZero();
		int points_outside = 0;

		Eigen::VectorXd dblA;
		igl::doublearea(F_curr, F.F, dblA);

		for (int i = 0; i < F.F.rows(); ++i)
		{
			double face_area = 0.5 * dblA(i);

			for (const auto &bary : quad_bary)
			{

				p_quad.row(0) = bary(0) * F_curr.row(F.F(i, 0)) +
								bary(1) * F_curr.row(F.F(i, 1)) +
								bary(2) * F_curr.row(F.F(i, 2));

				tree.squared_distance(C_hat.V, C_hat.F, p_quad, sqrD, I_face, C_closest);
				double dist = std::sqrt(std::max(0.0, sqrD(0)));

				Eigen::RowVector3d dir = p_quad.row(0) - C_closest.row(0);
				double sign = 1.0;

				if (dist > 1e-7)
				{
					sign = (dir.dot(FN.row(I_face(0))) < 0) ? -1.0 : 1.0;
				}

				if (sign > 0)
					points_outside++;

				Eigen::RowVector3d g;
				if (dist > 1e-5)
				{
					g = sign * dir / dist;
				}
				else
				{
					g = -FN.row(I_face(0));
				}

				double w0 = quad_weight * bary(0);
				double w1 = quad_weight * bary(1);
				double w2 = quad_weight * bary(2);

				for (int d = 0; d < 3; ++d)
				{
					grad_Phi(F.F(i, 0), d) += w0 * g(d);
					grad_Phi(F.F(i, 1), d) += w1 * g(d);
					grad_Phi(F.F(i, 2), d) += w2 * g(d);
				}
			}
		}

		if (iter % 10 == 0 || points_outside == 0)
		{
			std::cout << "      Flow Iteration " << iter << " | Points outside: " << points_outside << "\n";
		}

		if (points_outside == 0)
			break;

		for (int i = 0; i < grad_Phi.rows(); ++i)
		{
			if (!grad_Phi.row(i).allFinite())
				grad_Phi.row(i).setZero();
		}

		F_curr -= m_params.flow_dt * grad_Phi;
		H.push_back(F_curr);
	}
	return H;
}

void NestedCages::build_ipc_mesh(const Eigen::MatrixXd &C_V, const Eigen::MatrixXi &C_F,
								 const Eigen::MatrixXd &F_V, const Eigen::MatrixXi &F_F,
								 Eigen::MatrixXd &combined_V, Eigen::MatrixXi &combined_E, Eigen::MatrixXi &combined_F)
{
	combined_V.resize(C_V.rows() + F_V.rows(), 3);
	combined_V << C_V, F_V;
	combined_F.resize(C_F.rows() + F_F.rows(), 3);
	combined_F << C_F, (F_F.array() + C_V.rows()).matrix();
	igl::edges(combined_F, combined_E);
}

#include <ipc/collisions/normal/normal_collisions.hpp>
#include <ipc/potentials/barrier_potential.hpp>

NestedCages::Mesh NestedCages::reinflate(std::vector<Eigen::MatrixXd> &H, const Mesh &C_hat, const Mesh &F_orig)
{
	std::cout << "      [INIT] Structuring Isolated IPC Pipeline..." << std::endl;
	Eigen::MatrixXd C_curr = C_hat.V;

	Eigen::MatrixXd F_curr = H.back();
	H.pop_back();

	for (int i = 0; i < C_curr.rows(); ++i)
	{
		if (!C_curr.row(i).allFinite())
			C_curr.row(i) = C_hat.V.row(i);
	}
	for (int i = 0; i < F_curr.rows(); ++i)
	{
		if (!F_curr.row(i).allFinite())
			F_curr.row(i) = F_orig.V.row(i);
	}

	Eigen::VectorXi FC;
	igl::facet_components(C_hat.F, FC);
	int num_components = (FC.size() > 0) ? FC.maxCoeff() + 1 : 0;

	std::vector<int> pinned_verts;
	if (num_components > 0)
	{
		std::vector<bool> comp_pinned(num_components, false);
		for (int i = 0; i < C_hat.F.rows(); ++i)
		{
			int comp = FC(i);
			if (!comp_pinned[comp])
			{
				pinned_verts.push_back(C_hat.F(i, 0));
				comp_pinned[comp] = true;
			}
		}
	}
	else
	{
		pinned_verts.push_back(0);
	}

	Eigen::VectorXi b(pinned_verts.size());
	for (size_t i = 0; i < pinned_verts.size(); ++i)
		b(i) = pinned_verts[i];
	Eigen::MatrixXd bc(b.size(), 3);

	igl::ARAPData arap_data;
	arap_data.max_iter = 1;
	igl::arap_precomputation(C_hat.V, C_hat.F, 3, b, arap_data);

	Eigen::MatrixXi C_E;
	igl::edges(C_hat.F, C_E);
	ipc::CollisionMesh collision_mesh(C_curr, C_E, C_hat.F);

	double avg_edge_len = 0.0;
	for (int i = 0; i < C_E.rows(); ++i)
	{
		avg_edge_len += (C_curr.row(C_E(i, 0)) - C_curr.row(C_E(i, 1))).norm();
	}
	if (C_E.rows() > 0)
		avg_edge_len /= C_E.rows();

	Eigen::Vector3d bbox_min = C_curr.colwise().minCoeff();
	Eigen::Vector3d bbox_max = C_curr.colwise().maxCoeff();
	double max_allowable_disp = std::min((bbox_max - bbox_min).norm() * 0.01, avg_edge_len * 0.25);
	if (max_allowable_disp < 1e-5)
		max_allowable_disp = 1e-5;

	int total_frames = H.size();
	int current_frame = 0;

	Eigen::MatrixXd C_arap_guess(C_curr.rows(), 3);
	Eigen::MatrixXd U_C_arap(C_curr.rows(), 3);
	Eigen::MatrixXd U_C_repel(C_curr.rows(), 3);
	Eigen::MatrixXd U_C_ipc(C_curr.rows(), 3);
	Eigen::MatrixXd U_C(C_curr.rows(), 3);

	Eigen::MatrixXd C_VN, C_FN;
	double dynamic_dhat = m_params.ipc_dhat;

	while (!H.empty())
	{
		current_frame++;
		std::cout << "      Re-inflating Frame " << current_frame << " / " << total_frames << "..." << std::endl;

		Eigen::MatrixXd F_target = std::move(H.back());
		H.pop_back();

		for (int i = 0; i < F_target.rows(); ++i)
		{
			if (!F_target.row(i).allFinite())
				F_target.row(i) = F_orig.V.row(i);
		}

		int substep = 0;

		while (substep < 2000)
		{
			Eigen::MatrixXd U_F_total = F_target - F_curr;
			double dist_to_target = U_F_total.rowwise().norm().maxCoeff();

			if (dist_to_target < 1e-6)
			{
				F_curr = F_target;
				break;
			}

			Eigen::MatrixXd U_F = U_F_total;
			if (dist_to_target > max_allowable_disp)
			{
				U_F *= (max_allowable_disp / dist_to_target);
			}

			for (int i = 0; i < b.size(); ++i)
				bc.row(i) = C_curr.row(b(i));
			C_arap_guess = C_curr;
			igl::arap_solve(bc, arap_data, C_arap_guess);

			bool arap_failed = false;
			for (int i = 0; i < C_arap_guess.rows(); ++i)
			{
				if (!C_arap_guess.row(i).allFinite())
				{
					arap_failed = true;
					break;
				}
			}
			if (arap_failed)
				U_C_arap.setZero();
			else
				U_C_arap.noalias() = C_arap_guess - C_curr;

			U_C_repel.setZero();
			igl::per_vertex_normals(C_curr, C_hat.F, C_VN);
			igl::per_face_normals(C_curr, C_hat.F, C_FN);

			igl::AABB<Eigen::MatrixXd, 3> f_tree;
			f_tree.init(F_curr, F_orig.F);
			igl::AABB<Eigen::MatrixXd, 3> c_tree;
			c_tree.init(C_curr, C_hat.F);

			Eigen::VectorXi I_face(1);
			Eigen::VectorXd sqrD(1);
			Eigen::RowVector3d closest;

			for (int i = 0; i < C_curr.rows(); ++i)
			{
				f_tree.squared_distance(F_curr, F_orig.F, C_curr.row(i), sqrD, I_face, closest);
				double dist = std::sqrt(std::max(0.0, sqrD(0)));
				if (dist < dynamic_dhat)
				{
					double strength = std::pow(dynamic_dhat - dist, 2) * 50000.0;
					Eigen::RowVector3d vn = C_VN.row(i);
					if (vn.allFinite() && vn.norm() > 1e-8)
					{
						U_C_repel.row(i) += vn.normalized() * strength;
					}
				}
			}

			for (int i = 0; i < F_curr.rows(); ++i)
			{
				c_tree.squared_distance(C_curr, C_hat.F, F_curr.row(i), sqrD, I_face, closest);
				double dist = std::sqrt(std::max(0.0, sqrD(0)));
				if (dist < dynamic_dhat)
				{
					double strength = std::pow(dynamic_dhat - dist, 2) * 50000.0;
					int f_idx = I_face(0);
					Eigen::RowVector3d fn = C_FN.row(f_idx);
					if (fn.allFinite() && fn.norm() > 1e-8)
					{
						Eigen::RowVector3d force = (fn.normalized() * strength) / 3.0;
						U_C_repel.row(C_hat.F(f_idx, 0)) += force;
						U_C_repel.row(C_hat.F(f_idx, 1)) += force;
						U_C_repel.row(C_hat.F(f_idx, 2)) += force;
					}
				}
			}

			U_C_ipc.setZero();
			ipc::NormalCollisions collisions;
			collisions.build(collision_mesh, C_curr, dynamic_dhat);

			ipc::BarrierPotential barrier_potential(dynamic_dhat);
			Eigen::VectorXd grad_B = barrier_potential.gradient(collisions, collision_mesh, C_curr);

			if (grad_B.size() == C_curr.rows() * 3 && grad_B.allFinite())
			{
				for (int i = 0; i < C_curr.rows(); ++i)
				{
					Eigen::RowVector3d g(-grad_B(i * 3 + 0), -grad_B(i * 3 + 1), -grad_B(i * 3 + 2));
					if (g.norm() > 1e-8)
					{
						U_C_ipc.row(i) = g.normalized() * max_allowable_disp;
					}
				}
			}

			U_C.setZero();
			for (int i = 0; i < C_curr.rows(); ++i)
			{

				if (U_C_ipc.row(i).norm() > 1e-8)
				{
					U_C.row(i) = U_C_ipc.row(i) + (U_C_arap.row(i) * 0.05);
				}
				else if (U_C_repel.row(i).norm() > 1e-8)
				{
					U_C.row(i) = U_C_repel.row(i) + (U_C_arap.row(i) * 0.05);
				}
				else
				{
					U_C.row(i) = U_C_arap.row(i) * m_params.reinflation_beta_init;
				}

				double v_norm = U_C.row(i).norm();
				if (v_norm > max_allowable_disp)
					U_C.row(i) *= (max_allowable_disp / v_norm);
			}

			Eigen::MatrixXd C_target_substep = C_curr + U_C;
			double max_step = ipc::compute_collision_free_stepsize(collision_mesh, C_curr, C_target_substep);
			if (!std::isfinite(max_step))
				max_step = 0.0;

			double step_fraction = max_step * 0.8;

			if (step_fraction < 1e-4)
			{
				dynamic_dhat *= 1.2;
				if (dynamic_dhat > m_params.ipc_dhat * 10.0)
					dynamic_dhat = m_params.ipc_dhat * 10.0;

				for (int i = 0; i < C_curr.rows(); ++i)
				{
					if (C_VN.row(i).allFinite() && C_VN.row(i).norm() > 1e-8)
					{
						C_curr.row(i) += C_VN.row(i).normalized() * (max_allowable_disp * 0.1);
					}
				}
			}
			else
			{
				if (step_fraction > 1.0)
					step_fraction = 1.0;

				C_curr += step_fraction * U_C;
				F_curr += step_fraction * U_F;

				dynamic_dhat = m_params.ipc_dhat;
			}

			substep++;
		}
	}

	std::cout << "      [FINAL POLISH] Ensuring strict cage clearance..." << std::endl;

	for (int iter = 0; iter < 50; ++iter)
	{
		igl::AABB<Eigen::MatrixXd, 3> final_tree;
		final_tree.init(C_curr, C_hat.F);

		Eigen::MatrixXd C_FN;
		igl::per_face_normals(C_curr, C_hat.F, C_FN);

		Eigen::VectorXi I_face(1);
		Eigen::VectorXd sqrD(1);
		Eigen::RowVector3d closest;

		bool needs_push = false;

		for (int i = 0; i < F_orig.V.rows(); ++i)
		{
			final_tree.squared_distance(C_curr, C_hat.F, F_orig.V.row(i), sqrD, I_face, closest);
			double dist = std::sqrt(std::max(0.0, sqrD(0)));

			if (dist < m_params.ipc_dhat * 1.5)
			{
				needs_push = true;
				int f_idx = I_face(0);
				Eigen::RowVector3d fn = C_FN.row(f_idx);

				if (fn.allFinite() && fn.norm() > 1e-8)
				{

					Eigen::RowVector3d push = (fn.normalized() * max_allowable_disp * 0.5) / 3.0;
					C_curr.row(C_hat.F(f_idx, 0)) += push;
					C_curr.row(C_hat.F(f_idx, 1)) += push;
					C_curr.row(C_hat.F(f_idx, 2)) += push;
				}
			}
		}

		if (!needs_push)
			break;
	}

	Mesh C_final;
	C_final.V = C_curr;
	C_final.F = C_hat.F;
	return C_final;
}