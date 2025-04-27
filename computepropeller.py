import numpy as np  
import openmdao.api as om
import pdb


class ComputePropeller(om.ExplicitComponent):
    """
    Compute the power required for a propeller.

    References
    [1] GA Guddmundsson. "General Aviation Aircraft Design".
    
    """

    def initialize(self):
        self.options.declare('nn', types=int)
        self.options.declare('pod_id', default=1)

    def setup(self):
        nn = self.options['nn']
        pod_id = self.options['pod_id']

        # Inputs    
        self.add_input(f'pod{pod_id}|blade_diam', val=np.ones(nn), desc='blade diameter', units='m')
        self.add_input(f'pod{pod_id}|hub_diam', val=np.ones(nn), desc='hub diameter', units='m')
        self.add_input('rho', val=np.ones(nn), desc='air density', units='kg/m**3')
        self.add_input(f'pod{pod_id}|thrust_req', val=np.ones(nn), desc='total aircraft thrust required', units='N')
        self.add_input('tas', val=np.ones(nn), desc='true airspeed', units='m/s')
        self.add_input(f'pod{pod_id}|eta_prop', val=np.ones(nn), desc='propeller efficiency', units=None) 

        # Outputs
        self.add_output(f'pod{pod_id}|pow_req', val=np.ones(nn), desc='power required per engine', units='W')

    def setup_partials(self):

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        # Unpack options    
        pod_id = self.options['pod_id']

        # Unpack inputs
        eta_prop = inputs[f'pod{pod_id}|eta_prop']
        blade_diam = inputs[f'pod{pod_id}|blade_diam']
        hub_diam = inputs[f'pod{pod_id}|hub_diam']
        rho = inputs['rho']
        thrust_req = inputs[f'pod{pod_id}|thrust_req']
        tas = inputs['tas']  

        diskarea = np.pi * ((blade_diam/2)**2 - (hub_diam/2)**2)

        # Compute induced airspeed [1] Eq 15-76
        v_ind =  0.5 * ( - tas + ( tas**2 + 2*thrust_req / (0.5 * rho * diskarea) ) **(0.5) )

        # Compute station 3 ktasocity [1] Eq 15-73
        v3 = tas + 2 * v_ind

        # Compute propulsive efficiency [1] Eq 15-77
        eta_prplsv =  2 / (1 + v3/ ( tas))

        # Compute the power required [1] Eq 15-75
        p_prplsv_unit = thrust_req * tas  + thrust_req ** 1.5/ np.sqrt(2 * rho * diskarea)
   
        # Compute the power required [1] Eq 15-78
        pow_req = p_prplsv_unit / eta_prop / eta_prplsv

        # Pack outputs
        outputs[f'pod{pod_id}|pow_req'] = pow_req

    def compute_partials(self, inputs, partials):  

        # Unpack options
        pod_id = self.options['pod_id']

        # Unpack inputs
        tas = inputs['tas']
        blade_diam = inputs[f'pod{pod_id}|blade_diam']
        hub_diam = inputs[f'pod{pod_id}|hub_diam']
        rho = inputs['rho']
        thrust_req = inputs[f'pod{pod_id}|thrust_req']
        eta_prop = inputs[f'pod{pod_id}|eta_prop']


        partials[f'pod{pod_id}|pow_req', f'pod{pod_id}|blade_diam'] = np.diag(- (2**(1/2)*blade_diam*rho*thrust_req**(3/2)*((tas**2 + (4*thrust_req)/(np.pi*rho*(blade_diam**2/4 - hub_diam**2/4)))**(1/2)/(2*tas) + 1/2))/(8*eta_prop*np.pi**(1/2)*(rho*(blade_diam**2/4 - hub_diam**2/4))**(3/2)) - (blade_diam*thrust_req*(tas*thrust_req + (2**(1/2)*thrust_req**(3/2))/(2*np.pi**(1/2)*(rho*(blade_diam**2/4 - hub_diam**2/4))**(1/2))))/(2*eta_prop*rho*tas*np.pi*(tas**2 + (4*thrust_req)/(np.pi*rho*(blade_diam**2/4 - hub_diam**2/4)))**(1/2)*(blade_diam**2/4 - hub_diam**2/4)**2))

        partials[f'pod{pod_id}|pow_req', f'pod{pod_id}|hub_diam'] = np.diag( (2**(1/2)*hub_diam*rho*thrust_req**(3/2)*((tas**2 + (4*thrust_req)/(np.pi*rho*(blade_diam**2/4 - hub_diam**2/4)))**(1/2)/(2*tas) + 1/2))/(8*eta_prop*np.pi**(1/2)*(rho*(blade_diam**2/4 - hub_diam**2/4))**(3/2)) + (hub_diam*thrust_req*(tas*thrust_req + (2**(1/2)*thrust_req**(3/2))/(2*np.pi**(1/2)*(rho*(blade_diam**2/4 - hub_diam**2/4))**(1/2))))/(2*eta_prop*rho*tas*np.pi*(tas**2 + (4*thrust_req)/(np.pi*rho*(blade_diam**2/4 - hub_diam**2/4)))**(1/2)*(blade_diam**2/4 - hub_diam**2/4)**2))

        partials[f'pod{pod_id}|pow_req', f'pod{pod_id}|thrust_req'] = np.diag(((tas**2 + (4*thrust_req)/(np.pi*rho*(blade_diam**2/4 - hub_diam**2/4)))**(1/2)/(2*tas) + 1/2)*(tas + (3*2**(1/2)*thrust_req**(1/2))/(4*np.pi**(1/2)*(rho*(blade_diam**2/4 - hub_diam**2/4))**(1/2))))/eta_prop + (tas*thrust_req + (2**(1/2)*thrust_req**(3/2))/(2*np.pi**(1/2)*(rho*(blade_diam**2/4 - hub_diam**2/4))**(1/2)))/(eta_prop*rho*tas*np.pi*(tas**2 + (4*thrust_req)/(np.pi*rho*(blade_diam**2/4 - hub_diam**2/4)))**(1/2)*(blade_diam**2/4 - hub_diam**2/4))

        partials[f'pod{pod_id}|pow_req', 'rho'] = - np.diag(2**(1/2)*thrust_req**(3/2)*((tas**2 + (4*thrust_req)/(np.pi*rho*(blade_diam**2/4 - hub_diam**2/4)))**(1/2)/(2*tas) + 1/2)*(blade_diam**2/4 - hub_diam**2/4))/(4*eta_prop*np.pi**(1/2)*(rho*(blade_diam**2/4 - hub_diam**2/4))**(3/2)) - (thrust_req*(tas*thrust_req + (2**(1/2)*thrust_req**(3/2))/(2*np.pi**(1/2)*(rho*(blade_diam**2/4 - hub_diam**2/4))**(1/2))))/(eta_prop*rho**2*tas*np.pi*(tas**2 + (4*thrust_req)/(np.pi*rho*(blade_diam**2/4 - hub_diam**2/4)))**(1/2)*(blade_diam**2/4 - hub_diam**2/4))

        partials[f'pod{pod_id}|pow_req', f'pod{pod_id}|eta_prop'] = -np.diag(((tas**2 + (4*thrust_req)/(np.pi*rho*(blade_diam**2/4 - hub_diam**2/4)))**(1/2)/(2*tas) + 1/2)*(tas*thrust_req + (2**(1/2)*thrust_req**(3/2))/(2*np.pi**(1/2)*(rho*(blade_diam**2/4 - hub_diam**2/4))**(1/2))))/eta_prop**2

        partials[f'pod{pod_id}|pow_req', 'tas'] =np.diag( (thrust_req*((tas**2 + (4*thrust_req)/(np.pi*rho*(blade_diam**2/4 - hub_diam**2/4)))**(1/2)/(2*tas) + 1/2))/eta_prop - ((tas*thrust_req + (2**(1/2)*thrust_req**(3/2))/(2*np.pi**(1/2)*(rho*(blade_diam**2/4 - hub_diam**2/4))**(1/2)))*((tas**2 + (4*thrust_req)/(np.pi*rho*(blade_diam**2/4 - hub_diam**2/4)))**(1/2)/(2*tas**2) - 1/(2*(tas**2 + (4*thrust_req)/(np.pi*rho*(blade_diam**2/4 - hub_diam**2/4)))**(1/2))))/eta_prop)
 




if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem(reports=False)
    model = p.model

    mtom_lb = 86000 - 100 
    mtom_kg = mtom_lb * 0.453592
    g = 9.806
    l_d = 16.5
    num_pods = 4

    thrust_total = mtom_kg * g / l_d
    thrust_pod = thrust_total / num_pods

    pod_id = 1
    nn = 1

    ivc = om.IndepVarComp()
    ivc.add_output(f'pod{pod_id}|blade_diam', 3.9, units='m')
    ivc.add_output(f'pod{pod_id}|hub_diam', 0.75, units='m')
    ivc.add_output('rho', 0.4, units='kg/m**3')
    ivc.add_output(f'pod{pod_id}|thrust_req', thrust_pod, units='N')
    ivc.add_output('tas', 275, units='kn')
    ivc.add_output(f'pod{pod_id}|eta_prop', 0.8 , units=None)

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('Propeller', ComputePropeller(nn = nn, pod_id = pod_id), promotes_inputs=['*'], promotes_outputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    om.n2(p)
    p.run_model()

    print('pow_req = ', p[f'pod{pod_id}|pow_req']/1000, 'kW')

    p.check_partials(compact_print=True)
