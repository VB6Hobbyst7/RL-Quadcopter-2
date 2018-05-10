import numpy as np
from physics_sim import PhysicsSim

class TaskHover():

    def __init__(   self,
                    positionStart=None,
                    positionStartNoise=2,
                    positionTarget=None,
                    nActionRepeats= 1,
                    runtime=10.,
                    factorPositionZ=0.2,
                    factorPositionXY=0.2,
                    factorAngles=0.2,
                    factorAngleRates=0.4,
                    angleNoise=0 ):

        # Internalize parameters.
        # Positions
        self.positionTarget = positionTarget if positionTarget is not None else np.array([0., 0., 10.])
        self.positionStart = positionStart if positionStart is not None else np.array([0., 0., 0.])
        # Reward factors.
        self.factorPositionZ = factorPositionZ
        self.factorPositionXY = factorPositionXY
        self.factorAngles = factorAngles
        self.factorAngleRates = factorAngleRates
        self.positionStartNoise = positionStartNoise
        self.angleNoise = angleNoise
        # Number of action repeats.
        # For each agent time step we step the simulation multiple times and stack the states.
        self.nActionRepeats = nActionRepeats

        # Action and state envelope.
        # Include target position, pose and angular velocities in the state.
        self.state_size = self.nActionRepeats * 12
        # Limit actions to a range around hovering.
        self.action_low = 395
        self.action_high = 420
        self.action_size = 4

        # Init initial conditions.
        # Rotation is randomized by +-0.01 degrees around all axes.
        init_pose = np.hstack((self.positionStart, np.array([0., 0., 0.])))
        # Initial velocity 0,0,0.
        init_velocities = np.array([0., 0., 0.])
        # Initial angular velocity 0,0,0.
        init_angle_velocities = np.array([0., 0., 0.])

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)

    def get_reward(self):
        return self.rewardPositionXY(self.sim.pose) + self.rewardPositionZ(self.sim.pose) + self.rewardAngles(self.sim.pose) + self.rewardAngleRates(self.sim.angular_v)

    def rewardPositionZ(self, pose):
        #cGauss = 2
        cExp = -0.3
        #yGauss = np.exp(-((pose[2]-self.positionTarget[2])**2/(2*cGauss**2)))
        yExp = np.exp(cExp * np.abs(self.positionTarget[2] - pose[2]))
        return self.factorPositionZ * yExp

    def rewardPositionXY(self, pose):
        a = .5
        return self.factorPositionXY * np.exp(-np.power(np.linalg.norm(self.positionTarget[:2] - pose[:2]), a))

    def rewardAngles(self, pose):
        x = np.max(  np.abs(  [a if a <= np.pi else 2*np.pi-a for a in pose[3:5]]  )   )
        return np.cos(x) * self.factorAngles

    def rewardAngleRates(self, angular_v):
        c=0.5
        x = np.max(np.abs(angular_v[0:2]))
        return (np.exp(-((x**2)/(2*c**2)))*2-1) * self.factorAngleRates

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.nActionRepeats):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.positionTarget)
            pose_all.append(self.sim.pose)
            pose_all.append(self.sim.angular_v)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset(positionNoise=self.positionStartNoise, angleNoise=self.angleNoise)
        state = np.concatenate([self.positionTarget, self.sim.pose, self.sim.angular_v] * self.nActionRepeats)
        return state
