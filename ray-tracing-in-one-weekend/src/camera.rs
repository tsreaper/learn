use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Color, Point3, Vec3, random_unit_vector};
use std::io::{BufWriter, Write, stdout};

pub struct Camera {
    aspect_ratio: f64,
    image_width: i32,
}

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

fn ray_color(ray: &Ray, world: &impl Hittable, depth: i32) -> Color {
    if depth < 0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    match world.hit(ray, 0.001, f64::INFINITY) {
        Some(record) => match record.material().scatter(ray, &record) {
            Some((attenuation, scattered)) => attenuation * ray_color(&scattered, world, depth - 1),
            None => Color::new(0.0, 0.0, 0.0),
        },
        None => {
            let unit_direction = ray.direction().unit_vector();
            let a = 0.5 * (unit_direction.y() + 1.0);
            (1.0 - a) * Color::new(1.0, 1.0, 1.0) + a * Color::new(0.5, 0.7, 1.0)
        }
    }
}

impl Camera {
    pub fn new(aspect_ratio: f64, image_width: i32) -> Self {
        Self {
            aspect_ratio,
            image_width,
        }
    }

    pub fn render(&self, world: &impl Hittable) {
        let aspect_ratio = self.aspect_ratio;
        let image_width = self.image_width;
        let image_height = ((image_width as f64 / aspect_ratio) as i32).max(1);
        let samples_per_pixel = 100;
        let max_depth = 50;

        let focal_length = 1.0;
        let viewport_height = 2.0;
        let viewport_width = viewport_height * image_width as f64 / image_height as f64;
        let camera_center = Point3::new(0.0, 0.0, 0.0);

        let viewport_u = Vec3::new(viewport_width, 0.0, 0.0);
        let viewport_v = Vec3::new(0.0, -viewport_height, 0.0);

        let pixel_delta_u = viewport_u / image_width as f64;
        let pixel_delta_v = viewport_v / image_height as f64;

        let viewport_upper_left =
            camera_center - Vec3::new(0.0, 0.0, focal_length) - viewport_u / 2.0 - viewport_v / 2.0;
        let pixel00_loc = viewport_upper_left + 0.5 * pixel_delta_u + 0.5 * pixel_delta_v;

        let mut out = BufWriter::new(stdout());
        writeln!(out, "P3\n{image_width} {image_height}\n255")
            .expect("Failed to write header to stdout");

        for row in 0..image_height {
            eprint!("\rScanlines remaining: {}    ", image_height - row);
            for col in 0..image_width {
                let pixel_center =
                    pixel00_loc + (col as f64 * pixel_delta_u) + (row as f64 * pixel_delta_v);
                let mut pixel_color = Color::new(0.0, 0.0, 0.0);
                for _ in 0..samples_per_pixel {
                    let ray_direction = pixel_center
                        + rand::random_range(-0.5..0.5) * pixel_delta_u
                        + rand::random_range(-0.5..0.5) * pixel_delta_v
                        - camera_center;
                    let ray = Ray::new(camera_center, ray_direction);
                    pixel_color += ray_color(&ray, world, max_depth);
                }
                pixel_color /= samples_per_pixel as f64;
                write_color(&pixel_color, &mut out);
            }
        }

        eprintln!("\rDone.                         ");
    }
}
