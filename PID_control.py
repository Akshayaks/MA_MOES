import matplotlib.pyplot as plt
import numpy as np
import common
from ergodic_coverage import ErgCover
import jax.numpy as jnp
import ergodic_metric
from utils import *

def fDiffDrive(x0, u):
	"""
	x0 = (x,y,theta)
	u = (v,w), vx = vy = v
	"""
	# x = x0 + np.array([np.cos(x0[2])*np.abs(u[0]), np.sin(x0[2])*np.abs(u[0]), u[1]])
	u = 0.3*np.tanh(u) #Limit the maximum velocity to 1
	x = x0 + np.array([np.cos(x0[2])*np.abs(u[0]), np.sin(x0[2])*np.abs(u[0]), 10*u[1]])
	return x

def follow(pos,Xref,Uref):
    print("Going to follow the trajectory")
    #breakpoint()
    Kp = np.array([0.95,0.1])
    # Kd = 1.5
    curr_pos = np.array(pos)
    curr_vel = np.array([0,0])
    actual_tj = []
    p_err = []
    v_err = []
    for i in range(len(Xref)):
        pos_error = np.linalg.norm(curr_pos[:2] - Xref[i][:2])
        theta_error = curr_pos[2] - Xref[i][2]
        vel_error = curr_vel - Uref[i] #np.array([Uref[i][0], Uref[i][0], Uref[i][1]])
        p_err.append(np.linalg.norm(pos_error))
        v_err.append(np.linalg.norm(vel_error))
        # print("Position error: ", p_err[-1])
        # print("Velocity error: ", v_err[-1])
        # #breakpoint()
        curr_vel = -np.multiply(Kp,[pos_error,theta_error]) #- Kd*vel_error
        # print("Current vel: ", curr_vel)
        # print("Reference velocity: ", Uref[i])
        # v = curr_vel/np.cos(curr_pos[2])  #v = vx/cos(theta)
        curr_pos = fDiffDrive(curr_pos,curr_vel)
        actual_tj.append(curr_pos[:2])
    return p_err, v_err, np.array(actual_tj)
        

def simple_EC():
    n_agents = 1
    n_scalar = 10

    pbm_file = "build_prob/random_maps/random_map_28.pickle"

    problem = common.LoadProblem(pbm_file, n_agents, pdf_list=True)

    problem.nA = 100

    start_pos = np.load("./start_pos_ang_random_4_agents.npy",allow_pickle=True)
    problem.s0 = start_pos.item().get("random_map_28.pickle")
    print("Agent start positions allotted:", problem.s0)

    #breakpoint()

    display_map(problem,problem.s0)

    trajectories = []
    actual_tj = []

    for i in range(len(problem.pdfs)):
        if i == 1:
            break
        for j in range(n_agents):
            print("problem.s0: ", problem.s0)
            print("Agent number: ", j)
            print("Start position: ", problem.s0[j*3:j*3+3])
            #breakpoint()
            pdf = problem.pdfs[0]*0.5 + problem.pdfs[3]*0.5
            # for k in range(len(problem.pdfs)):
            #     pdf += problem.pdfs[k]
            pdf = jnp.asarray(pdf.flatten())
            # EC = ergodic_metric.ErgCalc(pdf,1,problem.nA,n_scalar,problem.pix)

            control, erg, iters = ErgCover(pdf, 1, problem.nA, problem.s0[j*3:j*3+3], n_scalar, problem.pix, 500, False, None, stop_eps=-1,grad_criterion=True)

            print("ErgCover done")
            print("problem s0: ", problem.s0)
            #breakpoint()
            _, tj = ergodic_metric.GetTrajXYTheta(control, problem.s0[j*3:j*3+3])
            print("Length of traj and controls: ", len(tj),len(control))
            trajectories.append(tj)

            problem.pdfs = [problem.pdfs[0]*0.5 + problem.pdfs[3]*0.5]
            # problem.s0 = problem.s0[j*3:j*3+3]

            # np.save("sample_traj.npy",tj)
            # np.save("sample_control.npy",control)

            # tj = np.load("sample_traj.npy")
            # control = np.load("sample_control.npy")

            display_map(problem,problem.s0,tj=[tj])

            Xref = tj
            Uref = control

            # #breakpoint()

            p_err, v_err, atj = follow(problem.s0[j*3:j*3+3],Xref,Uref)
            actual_tj.append(atj)

            print("Got the trajectory")
            print("Problem s0: ", problem.s0)
            #breakpoint()

            display_map(problem,problem.s0[j*3:j*3+3],tj=[atj],ref=[tj])

            timestep = np.arange(0,len(p_err))

            plt.plot(timestep,p_err)
            plt.plot(timestep,v_err)
            plt.legend(["position error", "velocity error"])
            plt.show() 

            # print("problem s0: ", problem.s0)
            # #breakpoint()  


if __name__ == "__main__":
    simple_EC()
