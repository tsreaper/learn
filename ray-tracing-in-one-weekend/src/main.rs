mod camera;
mod hittable;
mod material;
mod ray;
mod vec3;

use camera::Camera;
use hittable::HittableList;
use hittable::Sphere;
use material::{Lambertian, Metal};
use vec3::{Color, Point3};

fn main() {
    let material_ground = Lambertian::new(Color::new(0.8, 0.8, 0.0));
    let sphere_ground = Sphere::new(Point3::new(0.0, -100.5, -1.0), 100.0, material_ground);

    let material_center = Lambertian::new(Color::new(0.1, 0.2, 0.5));
    let sphere_center = Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5, material_center);

    let material_left = Metal::new(Color::new(0.8, 0.8, 0.8), 0.3);
    let sphere_left = Sphere::new(Point3::new(-1.0, 0.0, -1.0), 0.5, material_left);

    let material_right = Metal::new(Color::new(0.8, 0.6, 0.2), 1.0);
    let sphere_right = Sphere::new(Point3::new(1.0, 0.0, -1.0), 0.5, material_right);

    let mut world = HittableList::new();
    world.add(sphere_ground);
    world.add(sphere_center);
    world.add(sphere_left);
    world.add(sphere_right);

    let camera = Camera::new(16.0 / 9.0, 800);
    camera.render(&world);
}
