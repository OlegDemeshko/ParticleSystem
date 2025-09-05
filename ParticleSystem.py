# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ParticleSystem:
    def __init__(self, Lx=5, Ly=5, dt=0.01, vmax=0.1, grid=(4, 4)): # при L ~3 система плавиться, при ~5 пливе, при ~8 більш стійука
        self.Lx, self.Ly = Lx, Ly
        self.dt = dt
        self.dt2 = dt * dt
        self.vmax = vmax
        self.rows, self.cols = grid
        self.N = self.rows * self.cols

        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)
        self.vx = np.zeros(self.N)
        self.vy = np.zeros(self.N)

        self.initialize_particles()
        self.ax, self.ay, _ = self.compute_forces()

    def initialize_particles(self):
        dx = self.Lx / self.cols
        dy = self.Ly / self.rows

        for i, (row, col) in enumerate(np.ndindex((self.rows, self.cols))):
            self.x[i] = dx * (col + 0.5)
            self.y[i] = dy * (row + 0.5)
            self.vx[i] = self.vmax * (2 * np.random.rand() - 1)
            self.vy[i] = self.vmax * (2 * np.random.rand() - 1) # роміщення частинок в комірках та їх швидкості

        # Центрування імпульсу
        self.vx -= np.mean(self.vx)
        self.vy -= np.mean(self.vy)

    def compute_forces(self):
        ax = np.zeros(self.N)
        ay = np.zeros(self.N)
        pe = 0.0

        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                dx, dy = self.periodic_delta(self.x[i] - self.x[j], self.y[i] - self.y[j])
                r = np.sqrt(dx ** 2 + dy ** 2 + 1e-5)
                force, potential = self.lennard_jones(r)
                fx = force * dx
                fy = force * dy

                ax[i] += fx
                ay[i] += fy
                ax[j] -= fx
                ay[j] -= fy
                pe += potential
        return ax, ay, pe

    def lennard_jones(self, r):
        inv_r = 1.0 / r
        inv_r6 = inv_r ** 6
        force_mag = 24 * inv_r * inv_r6 * (2 * inv_r6 - 1) / r
        potential = 4 * inv_r6 * (inv_r6 - 1)
        return force_mag, potential

    def periodic_delta(self, dx, dy):  # частинки з великою швидкістю її втрачають
        if abs(dx) > 0.5 * self.Lx:
            dx -= np.sign(dx) * self.Lx
        if abs(dy) > 0.5 * self.Ly:
            dy -= np.sign(dy) * self.Ly
        return dx, dy

    def kinetic_energy(self):
        return 0.5 * np.sum(self.vx ** 2 + self.vy ** 2)

    def verlet_step(self):
        self.x += self.vx * self.dt + 0.5 * self.ax * self.dt2
        self.y += self.vy * self.dt + 0.5 * self.ay * self.dt2 # нове положення
        self.x %= self.Lx
        self.y %= self.Ly # щоб не виходили за межі області

        self.vx += 0.5 * self.ax * self.dt
        self.vy += 0.5 * self.ay * self.dt
        self.ax, self.ay, pe = self.compute_forces()
        self.vx += 0.5 * self.ax * self.dt
        self.vy += 0.5 * self.ay * self.dt

        ke = self.kinetic_energy()
        return ke, pe

    def compute_pressure(self, temperature):
        virial = 0
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                dx, dy = self.periodic_delta(self.x[i] - self.x[j], self.y[i] - self.y[j])
                r = np.sqrt(dx ** 2 + dy ** 2 + 1e-5)
                force, _ = self.lennard_jones(r)
                virial += force * (dx * dx + dy * dy) / r
        return (self.N * temperature + 0.5 * virial) / (self.Lx * self.Ly)
        
def animation(steps=500):
    ps = ParticleSystem()
    fig, ax = plt.subplots()
    scat = ax.scatter(ps.x, ps.y)
    ax.set_xlim(0, ps.Lx)
    ax.set_ylim(0, ps.Ly)

    def update(frame):
        ke, pe = ps.verlet_step()
        scat.set_offsets(np.c_[ps.x, ps.y])
        return scat,

    ani = FuncAnimation(fig, update, frames=steps, interval=50)
    plt.show()

animation()
