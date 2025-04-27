"""
Hybrid-Electric Propulsion Pod Group
This group models a hybrid propulsion system with:
- One gas turbine engine
- Two electric motors
- One propeller
All components are connected through a gearbox to drive the propeller
"""

import openmdao.api as om
import numpy as np
import pdb 
import matplotlib.pyplot as plt

from computepropeller import ComputePropeller
from computeamos import ComputeAtmos


class ComputeTrueAirspeed(om.ExplicitComponent):
    """
    Computes true airspeed (TAS) from equivalent airspeed (EAS)
    TAS = EAS * sqrt(rho_0/rho) where rho_0 is sea level density
    """
    
    def initialize(self):
        self.options.declare('nn', types=int)
        self.options.declare('rho_0', default=1.225)
        
    def setup(self):
        nn = self.options['nn']
        
        # Inputs
        self.add_input('eas', val=np.ones(nn), units='m/s',
                       desc='Equivalent airspeed')
        self.add_input('rho', val=np.ones(nn), units='kg/m**3',
                       desc='Air density')
        
        # Outputs
        self.add_output('tas', val=np.ones(nn), units='m/s',
                        desc='True airspeed')
        
        
        # Declare partials
        self.declare_partials('*', '*')
        
    def compute(self, inputs, outputs):
        eas = inputs['eas']
        rho = inputs['rho']

        rho_0 = self.options['rho_0']
        
        # TAS = EAS * sqrt(rho_0/rho)
        outputs['tas'] = eas * np.sqrt(rho_0 / rho)
        
    def compute_partials(self, inputs, partials):
        eas = inputs['eas']
        rho = inputs['rho']

        rho_0 = self.options['rho_0']
        
        # d(TAS)/d(EAS) = sqrt(rho_0/rho)
        partials['tas', 'eas'] = np.diag(np.sqrt(rho_0 / rho))
        
        # d(TAS)/d(rho) = -0.5 * eas * rho_0 * rho^(-3/2)
        partials['tas', 'rho'] = np.diag(-0.5 * eas * rho_0 * rho**(-1.5))


class ComputePodPower(om.ExplicitComponent):
    """
    Computes power distribution based on gas turbine throttle ratio.
    GT ratio > 1: Generation mode (excess GT power charges batteries)
    GT ratio < 1: Hybrid mode (electric motors provide supplemental power)
    """
    
    def initialize(self):
        self.options.declare('nn', types=int)
        self.options.declare('pod_id', default=1)

    def setup(self):
        nn = self.options['nn']
        pod_id = self.options['pod_id']
        
        # Inputs
        self.add_input(f'pod{pod_id}|gt_throttle_ratio', val=np.ones(nn), 
                      desc='Gas turbine power out / Power required')
        self.add_input(f'pod{pod_id}|pow_req', val=np.ones(nn), units='W',
                      desc='Total power required by pod')
        self.add_input(f'pod{pod_id}|max_motor_pow', val=np.ones(nn), units='W',
                      desc='Maximum power per motor')
        self.add_input(f'pod{pod_id}|max_gt_pow', val=np.ones(nn), units='W',
                      desc='Maximum gas turbine pow')
        
        # Outputs
        self.add_output(f'pod{pod_id}|gt_pow', val=np.ones(nn), units='W')
        self.add_output(f'pod{pod_id}|e_pow', val=np.ones(nn), units='W',
                       desc='Positive: motors providing power, Negative: battery charging')
        self.add_output(f'pod{pod_id}|hy_ratio', val=np.ones(nn),
                       desc='Fraction of power from electric (0 in generation mode)')
        self.add_output(f'pod{pod_id}|motor_throttle', val=np.ones(nn))
        self.add_output(f'pod{pod_id}|gt_throttle', val=np.ones(nn))

    def setup_partials(self):

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        # Unpack options
        pod_id = self.options['pod_id']

        # Unpack inputs
        max_motor_pow = inputs[f'pod{pod_id}|max_motor_pow']
        max_gt_pow = inputs[f'pod{pod_id}|max_gt_pow']
        gt_pod_ratio = inputs[f'pod{pod_id}|gt_throttle_ratio']
        power_req = inputs[f'pod{pod_id}|pow_req']

        gt_pow = gt_pod_ratio * power_req
        gt_throttle = gt_pow / max_gt_pow
        e_pow = power_req - gt_pow
        hy_ratio = e_pow / power_req
        motor_pow = e_pow / 2
        motor_throttle = motor_pow / max_motor_pow

        # Pack outputs
        outputs[f'pod{pod_id}|gt_pow'] = gt_pow
        outputs[f'pod{pod_id}|gt_throttle'] = gt_throttle
        outputs[f'pod{pod_id}|e_pow'] = e_pow
        outputs[f'pod{pod_id}|hy_ratio'] = hy_ratio
        outputs[f'pod{pod_id}|motor_throttle'] = motor_throttle

    def compute_partials(self, inputs, partials):

        # Unpack inputs
        pod_id = self.options['pod_id']
        nn = self.options['nn']

        max_motor_pow = inputs[f'pod{pod_id}|max_motor_pow']
        max_gt_pow = inputs[f'pod{pod_id}|max_gt_pow']
        gt_pod_ratio = inputs[f'pod{pod_id}|gt_throttle_ratio']
        power_req = inputs[f'pod{pod_id}|pow_req']

        # Compute partials
        partials[f'pod{pod_id}|gt_pow', f'pod{pod_id}|gt_throttle_ratio'] = np.diag(power_req)
        partials[f'pod{pod_id}|gt_pow', f'pod{pod_id}|pow_req'] = np.diag(gt_pod_ratio)
        partials[f'pod{pod_id}|gt_pow', f'pod{pod_id}|max_gt_pow'] = np.diag(np.zeros(nn))
        partials[f'pod{pod_id}|gt_pow', f'pod{pod_id}|max_motor_pow'] = np.diag(np.zeros(nn))

        partials[f'pod{pod_id}|gt_throttle', f'pod{pod_id}|gt_throttle_ratio'] = np.diag(power_req / max_gt_pow)
        partials[f'pod{pod_id}|gt_throttle', f'pod{pod_id}|pow_req'] = np.diag(gt_pod_ratio / max_gt_pow)
        partials[f'pod{pod_id}|gt_throttle', f'pod{pod_id}|max_gt_pow'] = np.diag(-gt_pod_ratio * power_req / (max_gt_pow**2))
        partials[f'pod{pod_id}|gt_throttle', f'pod{pod_id}|max_motor_pow'] = np.diag(np.zeros(nn))

        partials[f'pod{pod_id}|e_pow', f'pod{pod_id}|gt_throttle_ratio'] = np.diag(-power_req)
        partials[f'pod{pod_id}|e_pow', f'pod{pod_id}|pow_req'] = np.diag(1 - gt_pod_ratio)
        partials[f'pod{pod_id}|e_pow', f'pod{pod_id}|max_gt_pow'] = np.diag(np.zeros(nn))
        partials[f'pod{pod_id}|e_pow', f'pod{pod_id}|max_motor_pow'] = np.diag(np.zeros(nn))

        partials[f'pod{pod_id}|hy_ratio', f'pod{pod_id}|gt_throttle_ratio'] = np.diag(-np.ones_like(gt_pod_ratio))
        partials[f'pod{pod_id}|hy_ratio', f'pod{pod_id}|pow_req'] = np.diag(-(power_req - gt_pod_ratio*power_req)/power_req**2 - (gt_pod_ratio - 1)/power_req)
        partials[f'pod{pod_id}|hy_ratio', f'pod{pod_id}|max_gt_pow'] = np.diag(np.zeros(nn))
        partials[f'pod{pod_id}|hy_ratio', f'pod{pod_id}|max_motor_pow'] = np.diag(np.zeros(nn))

        partials[f'pod{pod_id}|motor_throttle', f'pod{pod_id}|gt_throttle_ratio'] = np.diag(-power_req/(2*max_motor_pow))
        partials[f'pod{pod_id}|motor_throttle', f'pod{pod_id}|pow_req'] = np.diag(-(gt_pod_ratio/2 - 1/2)/max_motor_pow)
        partials[f'pod{pod_id}|motor_throttle', f'pod{pod_id}|max_gt_pow'] = np.diag(np.zeros(nn))
        partials[f'pod{pod_id}|motor_throttle', f'pod{pod_id}|max_motor_pow'] = np.diag(-(power_req/2 - (gt_pod_ratio*power_req)/2)/max_motor_pow**2)


class ComputePodThrust(om.Group):
    """Group that contains thrust produced by a single pod"""
    
    def initialize(self):
        self.options.declare('nn', types=int)
        self.options.declare('pod_id', default=1)

    def setup(self):
        nn = self.options['nn']
        pod_id = self.options['pod_id']

        
        self.add_subsystem(f'pod{pod_id}_prop',
                          ComputePropeller(nn=nn,
                                    pod_id=pod_id),
                          promotes_inputs=['*'], promotes_outputs=['*'])
        
        self.add_subsystem(f'pod{pod_id}_power',
                          ComputePodPower(nn=nn,
                                          pod_id=pod_id),
                          promotes_inputs=['*'], promotes_outputs=['*'])


class GroupPods(om.Group):
    """Group assembling all pods"""

    def initialize(self):
        self.options.declare('nn', types=int)

    def setup(self):
        nn = self.options['nn']

        
        self.add_subsystem('atmos', 
                          ComputeAtmos(nn=nn),
                          promotes_inputs=['*'], promotes_outputs=['*'])
        
        # Add airspeed conversion component
        self.add_subsystem('airspeed',
                          ComputeTrueAirspeed(nn=nn),
                          promotes_inputs=['*'], 
                          promotes_outputs=['*'])

        self.add_subsystem(f'pod{1}',
                          ComputePodThrust(nn=nn,
                                   pod_id=1),
                          promotes_inputs=['*'], promotes_outputs=['*'])
        
        self.add_subsystem(f'pod{2}',
                    ComputePodThrust(nn=nn,
                            pod_id=2),
                          promotes_inputs=['*'], promotes_outputs=['*'])


        self.add_subsystem(f'pod{3}',
                    ComputePodThrust(nn=nn,
                            pod_id=3),
                          promotes_inputs=['*'], promotes_outputs=['*'])


        self.add_subsystem(f'pod{4}',
                    ComputePodThrust(nn=nn,
                            pod_id=4),
                          promotes_inputs=['*'], promotes_outputs=['*'])

        

        