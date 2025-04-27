import numpy as np  
import openmdao.api as om



class ComputeAtmos(om.ExplicitComponent):
    """
    Compute the thrust produced by a propeller.

    References
    [1]  NASA Earth Atmosphere Model ~ https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
    """

    def initialize(self):
        self.options.declare('nn', types=int)

    def setup(self):

        nn = self.options['nn']

        # Inputs    
        self.add_input('alt', val= np.ones(nn), desc='trajectory in the z-axis of earth fixed body frame', units='m')

        # Outputs
        self.add_output('rho', val= np.ones(nn), desc='air density', units='kg/m**3')
        self.add_output('temp', val= np.ones(nn), desc='air temperature', units='degC')
        self.add_output('pressure', val= np.ones(nn), desc='air pressure', units='Pa')

    def setup_partials(self):

        self.declare_partials('rho', 'alt')
        self.declare_partials('temp', 'alt')
        self.declare_partials('pressure', 'alt')

    def compute(self, inputs, outputs):

        # Unpack inputs 
        alt = inputs['alt'] #  altitude (m)
    
        # Temperature (degC)
        temp = 15.04 - 0.00649 * alt

        # Pressure (kPa)
        pressure = 101.29*((temp+273.1)/288.08)**5.256
        
        # Density (kg/m**3)
        rho = pressure/(0.2869*(temp+273.1))

        outputs['rho'] = rho
        outputs['temp'] = temp
        outputs['pressure'] = pressure

    def compute_partials(self, inputs, partials):

        nn = self.options['nn'] 

        alt = inputs['alt']        

        partials['temp', 'alt'] = np.diag(np.ones(nn)*-0.00649)
        partials['pressure', 'alt'] =np.diag( -(4318934697*(14407/14404 - (649*alt)/28808000)**(532/125))/360100000000)
        partials['rho', 'alt'] = np.diag( (4318934697*(14407/14404 - (649*alt)/28808000)**(532/125))/(360100000000*((1861981*alt)/1000000000 - 41333683/500000)) + (18860005549*(14407/14404 - (649*alt)/28808000)**(657/125))/(100000000000*((1861981*alt)/1000000000 - 41333683/500000)**2))





if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem(reports=False)
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('alt'  , 300, units='m')

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeAtmos', ComputeAtmos(nn=1), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('rho = ', p['ComputeAtmos.rho'], 'kg/m**3')
    print('temp = ', p['ComputeAtmos.temp'], 'degC')
    print('pressure = ', p['ComputeAtmos.pressure'], 'Pa')

    p.check_partials(compact_print=True)


