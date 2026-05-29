mod camera;
mod hittable;
mod material;
mod ray;
mod vec3;

use std::io::{BufWriter, Write, stdout};
use std::sync::Arc;
use std::sync::atomic::AtomicI64;
use std::thread;

use camera::Camera;
use hittable::HittableList;
use hittable::Sphere;
use material::{Dieletric, Lambertian, Metal};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use vec3::{Color, Point3};

fn linear_to_gamma(x: f64) -> f64 {
    if x > 0.0 { x.sqrt() } else { 0.0 }
}

fn write_color(color: &Color, out: &mut impl Write) {
    let r = linear_to_gamma(color.x());
    let g = linear_to_gamma(color.y());
    let b = linear_to_gamma(color.z());

    let ir = (255.999 * r) as i32;
    let ig = (255.999 * g) as i32;
    let ib = (255.999 * b) as i32;
    writeln!(out, "{ir} {ig} {ib}").expect("Failed to write color to stdout");
}

#[allow(dead_code)]
fn render(samples_per_pixel: i32, counter: &AtomicI64) -> Vec<Vec<Color>> {
    let mut world = HittableList::new();

    let material_ground = Lambertian::new(Color::new(0.8, 0.8, 0.0));
    let sphere_ground = Sphere::new(Point3::new(0.0, -100.5, -1.0), 100.0, material_ground);
    world.add(sphere_ground);

    let material_center = Lambertian::new(Color::new(0.1, 0.2, 0.5));
    let sphere_center = Sphere::new(Point3::new(0.0, 0.0, -1.2), 0.5, material_center);
    world.add(sphere_center);

    let material_left = Dieletric::new(1.5);
    let sphere_left = Sphere::new(Point3::new(-1.0, 0.0, -1.0), 0.5, material_left);
    world.add(sphere_left);

    let material_left_bubble = Dieletric::new(1.0 / 1.5);
    let sphere_left_bubble = Sphere::new(Point3::new(-1.0, 0.0, -1.0), 0.4, material_left_bubble);
    world.add(sphere_left_bubble);

    let material_right = Metal::new(Color::new(0.8, 0.6, 0.2), 0.0);
    let sphere_right = Sphere::new(Point3::new(1.0, 0.0, -1.0), 0.5, material_right);
    world.add(sphere_right);

    let camera = Camera::new(16.0 / 9.0, 800, samples_per_pixel);
    camera.render(&world, counter)
}

#[allow(dead_code)]
fn render_large(samples_per_pixel: i32, counter: &AtomicI64) -> Vec<Vec<Color>> {
    // Build the world from a fixed seed so every thread generates the identical
    // scene; only the per-pixel sampling differs between threads.
    let mut rng = StdRng::seed_from_u64(0);
    let mut world = HittableList::new();

    let ground_material = Lambertian::new(Color::new(0.5, 0.5, 0.5));
    world.add(Sphere::new(
        Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        ground_material,
    ));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = rng.random_range(0.0..1.0);
            let center = Point3::new(
                a as f64 + 0.9 * rng.random_range(0.0..1.0),
                0.2,
                b as f64 + 0.9 * rng.random_range(0.0..1.0),
            );

            if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                if choose_mat < 0.8 {
                    // diffuse
                    let albedo = Color::new(
                        rng.random_range(0.0..1.0),
                        rng.random_range(0.0..1.0),
                        rng.random_range(0.0..1.0),
                    ) * Color::new(
                        rng.random_range(0.0..1.0),
                        rng.random_range(0.0..1.0),
                        rng.random_range(0.0..1.0),
                    );
                    world.add(Sphere::new(center, 0.2, Lambertian::new(albedo)));
                } else if choose_mat < 0.95 {
                    // metal
                    let albedo = Color::new(
                        rng.random_range(0.5..1.0),
                        rng.random_range(0.5..1.0),
                        rng.random_range(0.5..1.0),
                    );
                    let fuzz = rng.random_range(0.0..0.5);
                    world.add(Sphere::new(center, 0.2, Metal::new(albedo, fuzz)));
                } else {
                    // glass
                    world.add(Sphere::new(center, 0.2, Dieletric::new(1.5)));
                }
            }
        }
    }

    world.add(Sphere::new(
        Point3::new(0.0, 1.0, 0.0),
        1.0,
        Dieletric::new(1.5),
    ));
    world.add(Sphere::new(
        Point3::new(-4.0, 1.0, 0.0),
        1.0,
        Lambertian::new(Color::new(0.4, 0.2, 0.1)),
    ));
    world.add(Sphere::new(
        Point3::new(4.0, 1.0, 0.0),
        1.0,
        Metal::new(Color::new(0.7, 0.6, 0.5), 0.0),
    ));

    let camera = Camera::new(16.0 / 9.0, 400, samples_per_pixel);
    camera.render(&world, counter)
}

fn main() {
    let total_samples = 100;
    let parallelism = thread::available_parallelism().map_or(1, |n| n.get());
    let samples_per_pixel = (total_samples / parallelism as i32).max(1);

    let counter = Arc::new(AtomicI64::new(0));
    let handles: Vec<_> = (0..parallelism)
        .map(|_| {
            let counter = counter.clone();
            thread::spawn(move || render(samples_per_pixel, &counter))
        })
        .collect();

    let results: Vec<Vec<Vec<Color>>> = handles
        .into_iter()
        .map(|h| h.join().expect("render thread panicked"))
        .collect();
    eprintln!("\rDone                             ");

    let image_height = results[0].len();
    let image_width = results[0][0].len();

    let num_images = results.len() as f64;

    let mut out = BufWriter::new(stdout());
    writeln!(out, "P3\n{image_width} {image_height}\n255")
        .expect("Failed to write header to stdout");

    for row in 0..image_height {
        for col in 0..image_width {
            let mut pixel_color = Color::new(0.0, 0.0, 0.0);
            for result in results.iter() {
                pixel_color += result[row][col];
            }
            pixel_color /= num_images;
            write_color(&pixel_color, &mut out);
        }
    }
}
