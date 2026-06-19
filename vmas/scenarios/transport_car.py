#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import math

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    """Transport scenario with car-like (nonholonomic) agents.

    Agents use a kinematic bicycle model whose minimum turning radius exceeds
    the package side length, forcing them to plan their approach direction.
    Agents have a small box shape (approximate point contact with the package).

    Action space per agent: [speed_cmd, steering_angle_cmd]  (2-D)
    Observation per agent:
        agent pos (2) + agent vel (2) + heading (cos θ, sin θ) (2)
        + per package: pkg-goal disp (2) + pkg-agent disp (2)
                        + pkg vel (2) + on_goal flag (1)
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 2)
        self.n_packages = kwargs.pop("n_packages", 1)
        self.package_width = kwargs.pop("package_width", 0.15)
        self.package_length = kwargs.pop("package_length", 0.15)
        self.package_mass = kwargs.pop("package_mass", 25)

        # Car kinematics — default min_turning_radius = 2 × package side
        self.min_turning_radius = kwargs.pop(
            "min_turning_radius", 2.0 * self.package_length
        )
        assert self.min_turning_radius > max(self.package_length, self.package_width), (
            "min_turning_radius must be larger than the package side length"
        )
        self.max_speed = kwargs.pop("max_speed", 1.0)
        self.agent_length = kwargs.pop("agent_length", 0.10)  # wheelbase = l_f + l_r
        self.agent_width = kwargs.pop("agent_width", 0.05)
        # Heavier agent: (a) more pushing force (F = mass * v / dt),
        # (b) larger moment of inertia → far less spin from corner contacts.
        # With mass=1 (default), I ≈ 0.001 kg·m² and a corner hit at collision_force=500
        # produces ~270 rad/s² — enough for a 90° spin in one step.
        # mass=5 reduces that to ~54 rad/s² (I=0.005 kg·m²), keeping turns under ~15°.
        self.agent_mass = kwargs.pop("agent_mass", 5.0)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        l_f = self.agent_length / 2.0
        l_r = self.agent_length / 2.0
        # Ensure R_min > package side: max_steering_angle = atan(wheelbase / R_min)
        max_steering_angle = math.atan(self.agent_length / self.min_turning_radius)

        self.shaping_factor = 100
        self.world_semidim = 1

        # Conservative bounding radius for agent placement
        self._agent_circ_radius = math.sqrt(
            (self.agent_length / 2) ** 2 + (self.agent_width / 2) ** 2
        )

        world = World(
            batch_dim,
            device,
            substeps=10,
            collision_force=500,
            x_semidim=(
                self.world_semidim
                + 2 * self._agent_circ_radius
                + max(self.package_length, self.package_width)
            ),
            y_semidim=(
                self.world_semidim
                + 2 * self._agent_circ_radius
                + max(self.package_length, self.package_width)
            ),
        )

        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Box(length=self.agent_length, width=self.agent_width),
                mass=self.agent_mass,
                u_range=[self.max_speed, max_steering_angle],
                u_multiplier=[1.0, 1.0],
                max_speed=self.max_speed,
                dynamics=KinematicBicycle(
                    world,
                    width=self.agent_width,
                    l_f=l_f,
                    l_r=l_r,
                    max_steering_angle=max_steering_angle,
                    integration="rk4",
                ),
            )
            world.add_agent(agent)

        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=0.15),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)

        self.packages = []
        for i in range(self.n_packages):
            package = Landmark(
                name=f"package {i}",
                collide=True,
                movable=True,
                mass=self.package_mass,
                shape=Box(length=self.package_length, width=self.package_width),
                color=Color.RED,
            )
            package.goal = goal
            self.packages.append(package)
            world.add_landmark(package)

        return world

    def reset_world_at(self, env_index: int = None):
        # Spawn agents with enough clearance (use circumscribed radius)
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=2 * self._agent_circ_radius,
            x_bounds=(-self.world_semidim, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
        )

        # Randomize agent headings uniformly in [0, 2π)
        for agent in self.world.agents:
            if env_index is None:
                rot = (
                    torch.rand(self.world.batch_dim, 1, device=self.world.device)
                    * 2
                    * torch.pi
                )
            else:
                rot = (
                    torch.rand(1, device=self.world.device) * 2 * torch.pi
                )
            agent.set_rot(rot, batch_index=env_index)

        agent_occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            agent_occupied_positions = agent_occupied_positions[env_index].unsqueeze(0)

        goal = self.world.landmarks[0]
        ScenarioUtils.spawn_entities_randomly(
            [goal] + self.packages,
            self.world,
            env_index,
            min_dist_between_entities=max(
                package.shape.circumscribed_radius() + goal.shape.radius + 0.01
                for package in self.packages
            ),
            x_bounds=(-self.world_semidim, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
            occupied_positions=agent_occupied_positions,
        )

        for package in self.packages:
            package.on_goal = self.world.is_overlapping(package, package.goal)

            if env_index is None:
                package.global_shaping = (
                    torch.linalg.vector_norm(
                        package.state.pos - package.goal.state.pos, dim=1
                    )
                    * self.shaping_factor
                )
            else:
                package.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        package.state.pos[env_index] - package.goal.state.pos[env_index]
                    )
                    * self.shaping_factor
                )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )

            for package in self.packages:
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)
                package.color = torch.tensor(
                    Color.RED.value,
                    device=self.world.device,
                    dtype=torch.float32,
                ).repeat(self.world.batch_dim, 1)
                package.color[package.on_goal] = torch.tensor(
                    Color.GREEN.value,
                    device=self.world.device,
                    dtype=torch.float32,
                )

                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew[~package.on_goal] += (
                    package.global_shaping[~package.on_goal]
                    - package_shaping[~package.on_goal]
                )
                package.global_shaping = package_shaping

        return self.rew

    def observation(self, agent: Agent):
        theta = agent.state.rot  # [batch, 1]
        cos_h = torch.cos(theta)
        sin_h = torch.sin(theta)
        heading = torch.cat([cos_h, sin_h], dim=-1)  # [batch, 2]

        # Signed forward speed: positive = moving forward, negative = reversing.
        # Without this the policy would need to learn dot(vel, heading) itself,
        # which is non-trivial for a plain MLP.
        v_forward = (
            agent.state.vel[:, 0:1] * cos_h + agent.state.vel[:, 1:2] * sin_h
        )  # [batch, 1]

        package_obs = []
        for package in self.packages:
            package_obs.append(package.state.pos - package.goal.state.pos)
            package_obs.append(package.state.pos - agent.state.pos)
            package_obs.append(package.state.vel)
            package_obs.append(package.on_goal.float().unsqueeze(-1))

        return torch.cat(
            [
                agent.state.pos,  # 2
                agent.state.vel,  # 2
                heading,          # 2  (cos θ, sin θ)
                v_forward,        # 1  signed forward speed (negative = reversing)
                *package_obs,     # 7 per package
            ],
            dim=-1,
        )  # total: 7 + 7 * n_packages  (14 for n_packages=1)

    def done(self):
        return torch.all(
            torch.stack(
                [package.on_goal for package in self.packages],
                dim=1,
            ),
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
