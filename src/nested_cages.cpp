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

// using IPC to handle the collision math
#include <ipc/ipc.hpp>
#include <ipc/collision_mesh.hpp>

NestedCages::NestedCages(const Mesh &input_mesh, const Parameters &params)
	: m_input_mesh(input_mesh), m_params(params) {}

std::vector<NestedCages::Mesh> NestedCages::compute()
{
	std::vector<Mesh> layers;
	layers.push_back(m_input_mesh);

	// alg 1: main loop to build each layer
	for (int i = 1; i <= m_params.num_layers; ++i)
	{
		std::cout << "\n========================================\n";
		std::cout << " Computing Layer " << i << " / " << m_params.num_layers << "\n";
		std::cout << "========================================\n";

		Mesh M_prev = layers.back();

		std::cout << "  [1/3] Decimating coarse cage...\n";
		// step 1: paper just says use a black box decimator. libigl works fine.
		Mesh C_hat = decimate(M_prev, m_params.decimation_ratio);
		Mesh F = M_prev;

		std::cout << "  [2/3] Flowing fine mesh inside coarse cage...\n";
		// step 2 (alg 2): shrink F until it fits entirely inside C_hat
		std::vector<Eigen::MatrixXd> H = shrink(C_hat, F);

		std::cout << "  [3/3] Re-inflating with ARAP & IPC CCD (Frames: " << H.size() << ")...\n";
		// step 3 (alg 3): blow it back up but block intersections
		Mesh M_i = reinflate(H, C_hat, F);

		layers.push_back(M_i);

		// dump to obj
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
	// don't decimate below a tet (4 faces) or it crashes
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

	// fix nans from garbage triangles
	for (int i = 0; i < FN.rows(); ++i)
	{
		if (!FN.row(i).allFinite())
			FN.row(i) = Eigen::RowVector3d(1, 0, 0);
	}

	// paper uses 3-point quadrature for the integral
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
				// get 3d pos of the quad point
				p_quad.row(0) = bary(0) * F_curr.row(F.F(i, 0)) +
								bary(1) * F_curr.row(F.F(i, 1)) +
								bary(2) * F_curr.row(F.F(i, 2));

				tree.squared_distance(C_hat.V, C_hat.F, p_quad, sqrD, I_face, C_closest);
				double dist = std::sqrt(std::max(0.0, sqrD(0)));

				Eigen::RowVector3d dir = p_quad.row(0) - C_closest.row(0);
				double sign = 1.0;

				// check if inside using normals. paper warned about this but it's usually fine
				if (dist > 1e-7)
				{
					sign = (dir.dot(FN.row(I_face(0))) < 0) ? -1.0 : 1.0;
				}

				if (sign > 0)
					points_outside++;

				// grad of signed distance (eq 6)
				Eigen::RowVector3d g;
				if (dist > 1e-5)
				{
					g = sign * dir / dist;
				}
				else
				{
					g = -FN.row(I_face(0)); // fallback if touching
				}

				// push gradient to the 3 vertices (eq 4 & 5)
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

		// stop when it's fully inside
		if (points_outside == 0)
			break;

		for (int i = 0; i < grad_Phi.rows(); ++i)
		{
			if (!grad_Phi.row(i).allFinite())
				grad_Phi.row(i).setZero();
		}

		// euler step down the gradient
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

NestedCages::Mesh NestedCages::reinflate(std::vector<Eigen::MatrixXd> &H, const Mesh &C_hat, const Mesh &F_orig)
{
	Eigen::MatrixXd C_curr = C_hat.V;

	// play history backwards
	Eigen::MatrixXd F_curr = H.back();
	H.pop_back();

	// find a vertex to pin so ARAP doesn't float into space
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

	// setup arap to keep the cage looking nice (Esarap from paper)
	igl::ARAPData arap_data;
	arap_data.max_iter = 1;
	igl::arap_precomputation(C_hat.V, C_hat.F, 3, b, arap_data);

	// mash meshes together for ipc spatial hash
	Eigen::MatrixXi comb_E, comb_F;
	comb_F.resize(C_hat.F.rows() + F_orig.F.rows(), 3);
	comb_F << C_hat.F, (F_orig.F.array() + C_hat.V.rows()).matrix();
	igl::edges(comb_F, comb_E);

	Eigen::MatrixXd initial_comb_V(C_curr.rows() + F_curr.rows(), 3);
	initial_comb_V << C_curr, F_curr;

	// black box collision handler
	ipc::CollisionMesh collision_mesh(initial_comb_V, comb_E, comb_F);

	int total_frames = H.size();
	int current_frame = 0;

	Eigen::MatrixXd C_arap_guess(C_curr.rows(), 3);
	Eigen::MatrixXd U_C_arap(C_curr.rows(), 3);
	Eigen::MatrixXd U_C_barrier(C_curr.rows(), 3);
	Eigen::MatrixXd U_C(C_curr.rows(), 3);
	Eigen::MatrixXd U_F(F_curr.rows(), 3);
	Eigen::MatrixXd comb_V(C_curr.rows() + F_curr.rows(), 3);
	Eigen::MatrixXd comb_U(C_curr.rows() + F_curr.rows(), 3);
	Eigen::MatrixXd comb_V_target(C_curr.rows() + F_curr.rows(), 3);

	double avg_edge_len = 0.0;
	for (int i = 0; i < comb_E.rows(); ++i)
	{
		avg_edge_len += (initial_comb_V.row(comb_E(i, 0)) - initial_comb_V.row(comb_E(i, 1))).norm();
	}
	if (comb_E.rows() > 0)
		avg_edge_len /= comb_E.rows();

	Eigen::Vector3d bbox_min = C_curr.colwise().minCoeff();
	Eigen::Vector3d bbox_max = C_curr.colwise().maxCoeff();

	// strict bound so ipc doesn't run out of memory from crazy velocities
	double max_allowable_disp = std::min((bbox_max - bbox_min).norm() * 0.01, avg_edge_len * 0.25);
	if (max_allowable_disp < 1e-5)
		max_allowable_disp = 1e-5;

	Eigen::RowVector3d P_mat;
	Eigen::RowVector3d F_closest;
	Eigen::VectorXi I_face(1);
	Eigen::VectorXd sqrD(1);

	// reverse flow time!
	while (!H.empty())
	{
		current_frame++;
		if (current_frame % 5 == 0 || H.empty())
		{
			std::cout << "      Re-inflating Frame " << current_frame << " / " << total_frames << "...\n";
		}

		Eigen::MatrixXd F_target = std::move(H.back());
		H.pop_back();
		// fine mesh has "infinite mass", MUST hit its target
		U_F.noalias() = F_target - F_curr;

		// sanitize just in case so AABB doesn't crash
		for (int i = 0; i < F_curr.rows(); ++i)
		{
			if (!F_curr.row(i).allFinite())
				F_curr.row(i) = F_orig.V.row(i);
		}

		igl::AABB<Eigen::MatrixXd, 3> f_tree;
		f_tree.init(F_curr, F_orig.F);

		comb_V.topRows(C_curr.rows()) = C_curr;
		comb_V.bottomRows(F_curr.rows()) = F_curr;

		double beta = m_params.reinflation_beta_init;
		int bisection_count = 0;

		while (true)
		{
			for (int i = 0; i < b.size(); ++i)
				bc.row(i) = C_curr.row(b(i));

			C_arap_guess = C_curr;
			igl::arap_solve(bc, arap_data, C_arap_guess);

			for (int i = 0; i < C_arap_guess.rows(); ++i)
			{
				if (!C_arap_guess.row(i).allFinite())
					C_arap_guess.row(i) = C_curr.row(i);
			}

			U_C_arap.noalias() = C_arap_guess - C_curr;
			U_C_barrier.setZero();

			// hack: manual barrier force so IPC doesn't choke on CCD subdivision
			if (beta > 1e-4)
			{
				for (int i = 0; i < C_curr.rows(); ++i)
				{
					P_mat = C_curr.row(i);
					f_tree.squared_distance(F_curr, F_orig.F, P_mat, sqrD, I_face, F_closest);
					double dist = std::sqrt(std::max(0.0, sqrD(0)));

					if (dist < m_params.ipc_dhat && dist > 1e-8)
					{
						Eigen::RowVector3d dir = P_mat - F_closest;
						if (dir.norm() > 1e-8)
						{
							dir.normalize();
							// crank up strength to push it away early
							double strength = std::pow(m_params.ipc_dhat - dist, 2) * 10000.0;
							U_C_barrier.row(i) += strength * dir;
						}
					}
				}
			}

			for (int i = 0; i < U_C_arap.rows(); ++i)
				if (!U_C_arap.row(i).allFinite())
					U_C_arap.row(i).setZero();
			for (int i = 0; i < U_C_barrier.rows(); ++i)
				if (!U_C_barrier.row(i).allFinite())
					U_C_barrier.row(i).setZero();

			// combine forces (maintain shape + avoid collision)
			U_C.noalias() = beta * (U_C_arap + U_C_barrier);

			comb_U.topRows(C_curr.rows()) = U_C;
			comb_U.bottomRows(F_curr.rows()) = U_F;

			// clamp insane speeds so we don't phase through geometry
			for (int i = 0; i < comb_U.rows(); ++i)
			{
				if (!comb_U.row(i).allFinite())
					comb_U.row(i).setZero();

				double v_norm = comb_U.row(i).norm();
				if (std::isfinite(v_norm) && v_norm > max_allowable_disp)
				{
					comb_U.row(i) *= (max_allowable_disp / v_norm);
				}

				if (!comb_U.row(i).allFinite())
					comb_U.row(i).setZero();
			}

			U_C = comb_U.topRows(C_curr.rows());
			U_F = comb_U.bottomRows(F_curr.rows());

			comb_V_target.noalias() = comb_V + comb_U;

			// ask IPC how far we can actually move without intersecting
			double max_step = ipc::compute_collision_free_stepsize(collision_mesh, comb_V, comb_V_target);

			// bisection loop if we got stuck (like in the paper)
			if (max_step < 1.0)
			{
				beta *= 0.2;
				bisection_count++;

				if (bisection_count > 6)
				{
					double safe_step = max_step * 0.8;
					// stop float noise from queuing up forever
					if (safe_step < 1e-4)
						safe_step = 0.0;

					C_curr += safe_step * U_C;
					F_curr += safe_step * U_F;
					break;
				}
			}
			else
			{
				C_curr += U_C;
				F_curr += U_F;
				break;
			}
		}
	}

	Mesh C_final;
	C_final.V = C_curr;
	C_final.F = C_hat.F;
	return C_final;
}