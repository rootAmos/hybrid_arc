import numpy as np
import matplotlib.pyplot as plt
import pdb


class PlotData:

    def __init__(self, prob, len_clb, len_crz, gamma_climb, gamma_descent,gamma_mission, nn, num_pods):
        self.prob = prob
        self.len_clb = len_clb
        self.len_crz = len_crz
        self.gamma_climb = gamma_climb
        self.gamma_descent = gamma_descent
        self.gamma_mission = gamma_mission
        self.nn = nn
        self.num_pods = num_pods

        
    def plot_results(self, prob):

        # Unpack selft
        len_clb = self.len_clb
        len_crz = self.len_crz
        gamma_climb = self.gamma_climb
        gamma_descent = self.gamma_descent
        gamma_mission = self.gamma_mission
        nn = self.nn
        num_pods = self.num_pods

        # Compute Mission Time
        #pdb.set_trace()
        t0 = 0
        dz_clb_m = np.diff(prob.get_val('alt')[0:len_clb]) * 0.3048
        dz_clb_m = np.concatenate([[0], dz_clb_m])

        dt_clb_s = dz_clb_m / (prob.get_val('tas')[0:len_clb] * np.sin(gamma_climb))
        t_climb_s = t0 + np.cumsum(dt_clb_s)
        t_crz_s = np.linspace(t_climb_s[-1], t_climb_s[-1] + 2 * 60 * 60, len_crz)

        dz_des_m = np.diff(prob.get_val('alt')[len_clb + len_crz-1:-1]) * 0.3048
        dz_des_m = np.concatenate([ dz_des_m, [0]])

        dt_des_s = dz_des_m / (prob.get_val('tas')[len_clb + len_crz-1:-1] * np.sin(gamma_descent))
        t_descent_s = t_crz_s[-1] + np.cumsum(dt_des_s)

        t_mission_s = np.concatenate([t_climb_s, t_crz_s, t_descent_s])
        t_mission_min = t_mission_s / 60


        #print(f"Mission Time: {t_mission_min:.2f} minutes")


        # Create mission points array for x-axis
        mission_points = np.arange(nn)

        # Extract data for plotting
        altitude = prob.get_val('alt')
        tas = prob.get_val('tas')
        thrust = prob.get_val('pod1|thrust_req')
        pod_power = prob.get_val('pod1|pow_req') / 1000  # Convert to kW

        # Get power data for all pods
        gt_power = np.zeros((num_pods, nn))
        e_power = np.zeros((num_pods, nn))
        motor_throttle = np.zeros((num_pods, nn))
        gt_throttle = np.zeros((num_pods, nn))
        hybrid_ratio = np.zeros((num_pods, nn))

        for i in range(1, num_pods+1):
            gt_power[i-1, :] = prob.get_val(f'pod{i}|gt_pow') / 1000  # Convert to kW
            e_power[i-1, :] = prob.get_val(f'pod{i}|e_pow') / 1000    # Convert to kW
            motor_throttle[i-1, :] = prob.get_val(f'pod{i}|motor_throttle')
            gt_throttle[i-1, :] = prob.get_val(f'pod{i}|gt_throttle')
            hybrid_ratio[i-1, :] = prob.get_val(f'pod{i}|hy_ratio')

        # Calculate total GT and electric power across all pods
        total_gt_power = np.sum(gt_power, axis=0)
        total_e_power = np.sum(e_power, axis=0)

        # Set up for nicer plots
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['figure.figsize'] = [12, 10]

        # Graph 1: Multiple subplots
        fig1, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
        fig1.suptitle('Hybrid-Electric Propulsion System - Mission Profile', fontsize=18)

        # Altitude subplot
        axs[0].plot(t_mission_min, altitude, 'b-', linewidth=2)
        axs[0].set_ylabel('Altitude (ft)')
        #axs[0].set_title('Mission Altitude Profile')
        axs[0].grid(True)

        # TAS subplot
        axs[1].plot(t_mission_min, tas, 'g-', linewidth=2)
        axs[1].set_ylabel('TAS (m/s)')
        #axs[1].set_title('True Airspeed')
        axs[1].grid(True)

        # Climb angle subplot
        axs[2].plot(t_mission_min, np.degrees(gamma_mission), 'r-', linewidth=2)
        axs[2].set_ylabel('Climb Angle (deg)')
        #axs[2].set_title('Flight Path Angle')
        axs[2].grid(True)

        # Pod power subplot
        axs[3].plot(t_mission_min, pod_power, 'm-', linewidth=2)
        axs[3].set_ylabel('Power (kW)')
        #axs[3].set_title('Pod Power Required')
        axs[3].grid(True)

        # Thrust subplot
        axs[4].plot(t_mission_min, thrust, 'k-', linewidth=2)
        axs[4].set_ylabel('Thrust (N)')
        axs[4].set_xlabel('Mission Time (minutes)')
        #axs[4].set_title('Pod Thrust Required')
        axs[4].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Graph 2: Stacked area chart for power distribution
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.set_title('Total Power Distribution by Source', fontsize=18)

        # Create stacked area chart - properly handling negative e_power
        ax2.fill_between(t_mission_min, 0, total_gt_power, color='#FF7F0E', alpha=0.7, label='Gas Turbine Power')

        # Separate positive and negative electric power
        pos_mask = total_e_power >= 0
        neg_mask = total_e_power < 0

        # For positive electric power - stack above GT power
        if np.any(pos_mask):
            ax2.fill_between(t_mission_min, 
                            np.where(pos_mask, total_gt_power, 0), 
                            np.where(pos_mask, total_gt_power + total_e_power, 0), 
                            color='#1F77B4', alpha=0.7, label='Electric Motor Power')

        # For negative electric power - show below zero as battery charging
        if np.any(neg_mask):
            ax2.fill_between(t_mission_min, 
                            np.where(neg_mask, total_e_power, 0), 
                            0, 
                            color='#2CA02C', alpha=0.7, label='Battery Charging')

        # Add the total power line
        ax2.plot(t_mission_min, total_gt_power + total_e_power, 'k-', linewidth=2, label='Net Power')

        # Also add the GT power line for clarity
        ax2.plot(t_mission_min, total_gt_power, 'r--', linewidth=1.5, alpha=0.7)

        ax2.set_xlabel('Mission Time (minutes)')
        ax2.set_ylabel('Power (kW)')
        ax2.grid(True)
        ax2.legend(loc='upper right')

        plt.tight_layout()

        # Graph 3: Motor throttle, GT throttle, and hybridization ratio for each pod
        fig3, axs3 = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        fig3.suptitle('Throttle Settings and Hybridization Ratio by Pod', fontsize=18)

        # Flatten the 2x2 array for easier indexing
        axs3_flat = axs3.flatten()

        for i in range(num_pods):
            ax = axs3_flat[i]
            
            # Plot throttle settings and ratios for each pod
            ax.plot(t_mission_min, motor_throttle[i], 'b-', linewidth=2, label='Motor Throttle')
            ax.plot(t_mission_min, gt_throttle[i], 'r-', linewidth=2, label='GT Throttle')
            ax.plot(t_mission_min, hybrid_ratio[i], 'g--', linewidth=2, label='Hybridization Ratio')
            ax.plot(t_mission_min, prob.get_val(f'pod{i+1}|gt_throttle_ratio'), 'k:', linewidth=2, label='GT / Pod Power Ratio')
            
            ax.set_title(f'Pod {i+1}')
            ax.grid(True)
            ax.set_ylim(-0.2, 1.4)  # Adjust if needed based on your data
            
            # Only add legend to the first plot to avoid clutter
            if i == 0:
                ax.legend(loc='upper right')

        # Add common x and y labels
        fig3.text(0.5, 0.04, 'Mission Time (minutes)', ha='center', fontsize=14)
        #fig3.text(0.04, 0.5, 'Throttle / Ratio', va='center', rotation='vertical', fontsize=14)

        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

        # Graph 4: Individual pod power distribution
        fig4, axs4 = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        fig4.suptitle('Power Distribution by Pod', fontsize=18)

        # Flatten the 2x2 array for easier indexing
        axs4_flat = axs4.flatten()

        for i in range(num_pods):
            ax = axs4_flat[i]
            
            # Get this pod's power data
            pod_gt_power = gt_power[i, :]
            pod_e_power = e_power[i, :]
            
            # Create stacked area chart for this pod - properly handling negative e_power
            ax.fill_between(t_mission_min, 0, pod_gt_power, color='#FF7F0E', alpha=0.7, label='Gas Turbine')
            
            # Separate positive and negative electric power
            pos_mask = pod_e_power >= 0
            neg_mask = pod_e_power < 0
            
            # For positive electric power - stack above GT power
            if np.any(pos_mask):
                ax.fill_between(t_mission_min, 
                                np.where(pos_mask, pod_gt_power, 0), 
                                np.where(pos_mask, pod_gt_power + pod_e_power, 0), 
                                color='#1F77B4', alpha=0.7, label='Electric (Motor)')
            
            # For negative electric power - show below zero as battery charging
            if np.any(neg_mask):
                ax.fill_between(t_mission_min, 
                                np.where(neg_mask, pod_e_power, 0), 
                                0, 
                                color='#2CA02C', alpha=0.7, label='Battery Charging')
            
            # Add the total power line
            ax.plot(t_mission_min, pod_gt_power + pod_e_power, 'k-', linewidth=2, label='Net Power')
            
            # Also add the GT power line for clarity
            ax.plot(t_mission_min, pod_gt_power, 'r--', linewidth=1.5, alpha=0.7)
            
            ax.set_title(f'Pod {i+1}')
            ax.grid(True)
            
            # Only add legends to the first plot to avoid clutter
            if i == 0:
                ax.legend(loc='upper right')

        # Add common x and y labels
        fig4.text(0.5, 0.04, 'Mission Time (minutes)', ha='center', fontsize=14)
        fig4.text(0.04, 0.5, 'Power (kW)', va='center', rotation='vertical', fontsize=14)

        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

        # Show all plots
        plt.show()

