use std::ops::{Index, IndexMut};
use std::marker::PhantomData;
use std::mem::swap;

const N: usize = 20;
const SIZE: usize = (N + 2) * (N + 2);

struct Grid<T> {
    data: [T; SIZE],
}

fn idx(i: usize, j: usize) -> usize {
    j * (N + 2) + i
}

impl<T> Index<[usize; 2]> for Grid<T> {
    type Output = T;

    fn index(&self, i: [usize; 2]) -> &T {
        &self.data[idx(i[0], i[1])]
    }
}

impl<T> IndexMut<[usize; 2]> for Grid<T> {
    fn index_mut(&mut self, i: [usize; 2]) -> &mut T {
        &mut self.data[idx(i[0], i[1])]
    }
}

#[derive(Clone, Copy, Debug)]
enum Direction {
    X,
    Y,
}

impl Grid<f32> {
    fn add_source(&mut self, source: &Grid<f32>, dt: f32) {
        for (d, s) in self.data.iter_mut().zip(source.data.iter()) {
            *d += s * dt;
        }
    }

    fn diffuse(&mut self, prev: &Grid<f32>, diff: f32, dt: f32, clear: Option<Direction>) {
        let a = dt * diff * (N * N) as f32;

        for i in 0..20 {
            for (x, l, r, u, d, p) in self.backward(prev) {
                *x = (p + a * (l + r + u + d)) / (1.0 + 4.0 * a);
            }
            self.set_bnd(clear);
        }
    }

    fn advect(&mut self, prev: &Grid<f32>, u: &Grid<f32>, v: &Grid<f32>, dt: f32, clear: Option<Direction>)
    {
        let dt = dt * N as f32;

        for i in 1..N + 1 {
            for j in 1..N + 1 {
                let x = u.trace(i, j, Direction::X, dt);
                let i0 = x.floor() as usize;
                let i1 = i0 + 1;
                let s = x.fract();

                let y = v.trace(i, j, Direction::Y, dt);
                let j0 = y.floor() as usize;
                let j1 = j0 + 1;
                let t = y.fract();

                self[[i, j]] = t * (s * prev[[i0, j0]] + (1.0 - s) * prev[[i1, j0]]) +
                    (1.0 - t) * (s * prev[[i0, j1]] + (1.0 - s) * prev[[i1, j1]]);
            }
        }
        self.set_bnd(clear);
    }

    fn set_bnd(&mut self, clear: Option<Direction>) {
        for i in 1..N + 1 {
            let x_dir = if let Some(Direction::X) = clear {
                -1.0
            } else {
                1.0
            };
            self[[0, i]] = self[[1, i]] * x_dir;
            self[[N + 1, i]] = self[[N, i]] * x_dir;

            let y_dir = if let Some(Direction::Y) = clear {
                -1.0
            } else {
                1.0
            };
            self[[i, 0]] = self[[i, 1]] * y_dir;
            self[[i, N + 1]] = self[[i, N]] * y_dir;
        }

        self[[0, 0]] = 0.5 * (self[[0, 1]] + self[[1, 0]]);
        self[[N + 1, 0]] = 0.5 * (self[[N, 0]] + self[[N + 1, 1]]);
        self[[0, N + 1]] = 0.5 * (self[[0, N]] + self[[1, N + 1]]);
        self[[N + 1, N + 1]] = 0.5 * (self[[N, N + 1]] + self[[N + 1, N]]);
    }

    fn trace(&self, i: usize, j: usize, dir: Direction, dt: f32) -> f32 {
        let mut x = match dir {
            Direction::X => i,
            Direction::Y => j,
        } as f32;

        x -= self[[i, j]] * dt;

        if x < 0.5 {
            0.5
        } else {
            let n = N as f32 + 0.5;
            if x > n {
                n
            } else {
                x
            }
        }
    }

    fn backward<'a>(&'a mut self, prev: &'a Grid<f32>) -> Backward<'a> {
        let grid = (&mut self[[1, 1]]) as *mut f32;
        let prev = (&prev[[1, 1]]) as *const f32;
        Backward {
            grid,
            prev,
            i: 1,
            j: 1,
            _mark: PhantomData,
        }
    }
}

fn project(u: &mut Grid<f32>, v: &mut Grid<f32>, div_buf: &mut Grid<f32>, p_buf: &mut Grid<f32>) {
    let h = 1.0 / N as f32;

    for i in 1..N + 1 {
        for j in 1..N + 1 {
            div_buf[[i, j]] = -0.5 * h *
                (u[[i + 1, j]] - u[[i - 1, j]] + v[[i, j + 1]] - v[[i, j - 1]]);
            p_buf[[i, j]] = 0.0;
        }
    }
    div_buf.set_bnd(None);
    p_buf.set_bnd(None);

    for k in 0..20 {
        for (v, l, r, u, d, prev) in p_buf.backward(div_buf) {
            *v = (prev + l + r + u + d) / 4.0;
        }
        p_buf.set_bnd(None);
    }

    for i in 1..N + 1 {
        for j in 1..N + 1 {
            u[[i, j]] -= 0.5 * (p_buf[[i + 1, j]] - p_buf[[i - 1, j]]) / h;
            v[[i, j]] -= 0.5 * (p_buf[[i, j + 1]] - p_buf[[i, j - 1]]) / h;
        }
    }
    u.set_bnd(Some(Direction::X));
    v.set_bnd(Some(Direction::Y));
}

struct Backward<'a> {
    grid: *mut f32,
    prev: *const f32,
    i: usize,
    j: usize,
    _mark: PhantomData<&'a mut f32>
}
    
type Up = f32;
type Down = f32;
type Left = f32;
type Right = f32;

impl<'a> Iterator for Backward<'a> {
    type Item = (&'a mut f32, Left, Right, Up, Down, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.j > N {
            None
        } else {
            unsafe {
                let v = &mut *self.grid;
                let left = *self.grid.offset(-1);
                let right = *self.grid.offset(1);
                let up = *self.grid.offset(-(N as isize + 2));
                let down = *self.grid.offset(N as isize + 2);
                let prev = *self.prev;

                self.i += 1;
                if self.i > N {
                    self.i = 1;
                    self.j += 1;
                    self.grid = self.grid.offset(3);
                    self.prev = self.prev.offset(3);
                } else {
                    self.grid = self.grid.offset(1);
                    self.prev = self.prev.offset(1);
                }

                Some((v, left, right, up, down, prev))
            }
        }
    }
}

pub struct Simulation {
    dens0: Grid<f32>,
    dens1: Grid<f32>,
    diff: f32,

    u0: Grid<f32>,
    u1: Grid<f32>,
    v0: Grid<f32>,
    v1: Grid<f32>,
    visc: f32,
}

impl Simulation {
    fn dens_step(&mut self, dt: f32) {
        self.dens1.add_source(&self.dens0, dt);
        swap(&mut self.dens0, &mut self.dens1);
        self.dens1.diffuse(&self.dens0, self.diff, dt, None);
        swap(&mut self.dens0, &mut self.dens1);
        self.dens1.advect(&self.dens0, &self.u1, &self.v1, dt, None);
    }

    fn vel_step(&mut self, dt: f32) {
        self.u1.add_source(&self.u0, dt);
        self.v1.add_source(&self.v0, dt);

        swap(&mut self.u0, &mut self.u1);
        swap(&mut self.v0, &mut self.v1);
        self.u1.diffuse(&self.u0, self.visc, dt, Some(Direction::X));
        self.v1.diffuse(&self.v0, self.visc, dt, Some(Direction::Y));
        project(&mut self.u1, &mut self.v1, &mut self.u0, &mut self.v0);

        swap(&mut self.u0, &mut self.u1);
        swap(&mut self.v0, &mut self.v1);
        self.u1.advect(&self.u0, &self.u0, &self.v0, dt, Some(Direction::X));
        self.v1.advect(&self.v0, &self.u0, &self.v0, dt, Some(Direction::Y));
        project(&mut self.u1, &mut self.v1, &mut self.u0, &mut self.v0);
    }
}

