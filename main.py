import openmdao.api as om
import numpy as np
import pdb 
import matplotlib.pyplot as plt

from computepropeller import ComputePropeller
from computeamos import ComputeAtmos
from grouphybridpod import GroupPods
from plot_results import PlotData

"""Main function for executing the HybridPod analysis"""
import openmdao.api as om
import numpy as np

# Define number of analysis points
nn = 30  # Single point analysis

# Create the OpenMDAO problem
prob = om.Problem(reports=False)
# Define independent variables
ivc = om.IndepVarComp()

mtom_lb = 85000 
mtom_kg = mtom_lb * 0.453592
g = 9.806
l_d = 17
num_pods = 4

len_climb = int(np.floor(nn/2-1))
x = np.arange(len_climb)

x0 = 0.5 # final climb angle gradient
x1 = 9 # initial climb angle gradient
k = 3 # decay rate 

r = x1 - x0
climb_ang_grad =  x0 + r * np.exp(-k * x / len_climb)
climb_ang_rad = np.arctan(climb_ang_grad/100)   

descent_ang_grad = -5
descent_ang_rad = np.arctan(descent_ang_grad/100)

# Build Flight Path Angle profile       
gamma_climb = climb_ang_rad * np.ones(len_climb)
len_descent = len_climb
gamma_descent = descent_ang_rad * np.ones(len_descent)
gamma_cruise = 0 * np.ones(nn - len_climb - len_descent)
gamma_mission = np.concatenate([gamma_climb, gamma_cruise, gamma_descent])


# Build thrust profile
thrust_idle = 1000 * np.ones(nn)
thrust_calc = mtom_kg * g * (1/ l_d + np.sin(gamma_mission))
thrust_total = np.where(thrust_calc > thrust_idle, thrust_calc, thrust_idle)
thrust_pod = thrust_total / num_pods

# Add equivalent airspeed input
ivc.add_output('eas', val=190 * np.ones(nn), units='kn')

# Build GT Pod Power Ratio profile
x0 = 0.6 # minimum pod power ratio
x1 = 0.9 # maximum pod power ratio

r = x1 - x0 # range of pod power ratio
y = x0 + r * (np.log(x+1) / np.log(len_climb+1))

gt_throttle_ratio_clb = y # throttle ratio for climb
gt_throttle_ratio_des =  1.4 * np.ones(len_descent)
gt_throttle_ratio_cru =  np.ones(nn - len(gt_throttle_ratio_clb) - len(gt_throttle_ratio_des))
gt_throttle_ratio_mission = np.concatenate([gt_throttle_ratio_clb, gt_throttle_ratio_cru, gt_throttle_ratio_des])

ivc.add_output('gt_throttle_ratio_mission', val=gt_throttle_ratio_mission)

for i in range(num_pods):
    # Propeller parameters
    ivc.add_output(f'pod{i+1}|blade_diam', 3.9 * np.ones(nn), units='m')
    ivc.add_output(f'pod{i+1}|hub_diam', 0.75 * np.ones(nn), units='m')
    ivc.add_output(f'pod{i+1}|thrust_req', thrust_pod * np.ones(nn), units='N')
    ivc.add_output(f'pod{i+1}|eta_prop', 0.8 * np.ones(nn), units=None)

    # Pod power parameters
    ivc.add_output(f'pod{i+1}|gt_throttle_ratio', val=gt_throttle_ratio_mission)
    ivc.add_output(f'pod{i+1}|max_motor_pow', val=5e5 * np.ones(nn), units='W')
    ivc.add_output(f'pod{i+1}|max_gt_pow', val=2e6 * np.ones(nn), units='W')

# Flight conditions
# Build flight profile 
alt_climb = np.linspace(5000, 30000, int(np.floor(nn/2-1)))
alt_descent = np.linspace(30000, 1000, int(np.floor(nn/2-1)))
alt_cruise = np.ones(nn-len(alt_climb)-len(alt_descent)) * 30000

len_crz = len(alt_cruise)
len_des = len(alt_descent)
len_clb = len(alt_climb)

alt = np.concatenate([alt_climb, alt_cruise, alt_descent])



ivc.add_output('alt', val=alt, units='ft')

# Add IVC and HybridPod to the model
prob.model.add_subsystem('inputs', ivc, promotes_outputs=['*'])
prob.model.add_subsystem('hybrid_pod', 
                        GroupPods(nn=nn),
                        promotes_inputs=['*'], promotes_outputs=['*'])

# Setup the problem
prob.setup()

#om.n2(prob)
#prob.check_partials(compact_print=True)

# Run the analysis
prob.run_model()


"""

# Display key outputs
print("\n===== Hybrid-Electric Propulsion Pod Analysis Results =====")

# Display airspeed conversion results
print("\nAirspeed Conversion Results:")
for i in range(nn):
    alt = prob.get_val('alt')[i]
    eas = prob.get_val('eas')[i]
    tas = prob.get_val('tas')[i]
    rho = prob.get_val('rho')[i]
    print(f"  Alt: {alt:.0f} ft, EAS: {eas:.1f} kn, TAS: {tas:.1f} kn, Density: {rho:.5f} kg/m³")

for i in range(1, num_pods+1):
    print(f"\nPod {i} Results:")
    print(f"  Pod Thrust:        {prob.get_val(f'pod{i}|thrust_req')[0]:.2f} N")
    print(f"  Pod Power:         {prob.get_val(f'pod{i}|pow_req')[0]/1e3:.2f} kW")
    print(f"  GT Power:          {prob.get_val(f'pod{i}|gt_pow')[0]/1e3:.2f} kW")
    print(f"  Electric Power:    {prob.get_val(f'pod{i}|e_pow')[0]/1e3:.2f} kW")
    print(f"  Hybrid Ratio:      {prob.get_val(f'pod{i}|hy_ratio')[0]:.3f}")
    print(f"  Motor Throttle:    {prob.get_val(f'pod{i}|motor_throttle')[0]:.3f}")
    print(f"  GT Throttle:       {prob.get_val(f'pod{i}|gt_throttle')[0]:.3f}")
"""

plot_data = PlotData(prob, 
                     len_clb = len_clb, 
                     len_crz = len_crz, 
                     gamma_climb = gamma_climb, 
                     gamma_descent = gamma_descent,
                     gamma_mission = gamma_mission, 
                     nn = nn, 
                     num_pods = num_pods)
plot_data.plot_results(prob)
