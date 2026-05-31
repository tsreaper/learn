use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Color, Point3, Vec3, random_in_unit_disk};
use core::f64;
use std::sync::atomic::{AtomicI64, Ordering};

pub struct Camera {
    aspect_ratio: f64,
    image_width: i32,
    samples_per_pixel: i32,
}

fn ray_color(ray: &Ray, world: &impl Hittable, depth: i32) -> Color {
    if depth < 0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    match world.hit(ray, 0.001, f64::INFINITY) {
        Some((record, material)) => match material.scatter(ray, &record) {
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

fn degrees_to_radius(degree: f64) -> f64 {
    degree / 180.0 * f64::consts::PI
}

impl Camera {
    pub fn new(aspect_ratio: f64, image_width: i32, samples_per_pixel: i32) -> Self {
        Self {
            aspect_ratio,
            image_width,
            samples_per_pixel,
        }
    }

    pub fn render(
        &self,
        world: &impl Hittable,
        vfov: f64,
        look_from: Point3,
        look_at: Point3,
        vup: Vec3,
        defocus_angle: f64,
        focus_dist: f64,
        counter: &AtomicI64,
    ) -> Vec<Vec<Color>> {
        let aspect_ratio = self.aspect_ratio;
        let image_width = self.image_width;
        let image_height = ((image_width as f64 / aspect_ratio) as i32).max(1);
        let max_depth = 50;

        let viewport_height = 2.0 * f64::tan(degrees_to_radius(vfov) * 0.5) * focus_dist;
        let viewport_width = viewport_height * image_width as f64 / image_height as f64;

        let w = (look_from - look_at).unit_vector();
        let u = vup.cross(w).unit_vector();
        let v = w.cross(u);
        let viewport_u = viewport_width * u;
        let viewport_v = -viewport_height * v;
        let viewport_upper_left = look_from - focus_dist * w - viewport_u / 2.0 - viewport_v / 2.0;

        let defocus_radius = focus_dist * f64::tan(degrees_to_radius(defocus_angle) * 0.5);
        let defocus_disk_u = u * defocus_radius;
        let defocus_disk_v = v * defocus_radius;

        let pixel_delta_u = viewport_u / image_width as f64;
        let pixel_delta_v = viewport_v / image_height as f64;
        let pixel00_loc = viewport_upper_left + 0.5 * pixel_delta_u + 0.5 * pixel_delta_v;

        counter.fetch_add(image_height as i64, Ordering::Relaxed);
        let mut result: Vec<Vec<Color>> = Vec::new();
        for row in 0..image_height {
            let remaining = counter.fetch_sub(1, Ordering::Relaxed);
            eprint!("\rScanlines remaining: {remaining}    ");
            let mut row_result: Vec<Color> = Vec::new();
            for col in 0..image_width {
                let pixel_center =
                    pixel00_loc + (col as f64 * pixel_delta_u) + (row as f64 * pixel_delta_v);
                let mut pixel_color = Color::new(0.0, 0.0, 0.0);
                for _ in 0..self.samples_per_pixel {
                    let pixel_sample = pixel_center
                        + rand::random_range(-0.5..0.5) * pixel_delta_u
                        + rand::random_range(-0.5..0.5) * pixel_delta_v;
                    let ray_origin = if defocus_radius <= 0.0 {
                        look_from
                    } else {
                        let p: Vec3 = random_in_unit_disk();
                        look_from + defocus_disk_u * p.x() + defocus_disk_v * p.y()
                    };
                    let ray_direction = pixel_sample - ray_origin;
                    let ray = Ray::new(ray_origin, ray_direction);
                    pixel_color += ray_color(&ray, world, max_depth);
                }
                pixel_color /= self.samples_per_pixel as f64;
                row_result.push(pixel_color);
            }
            result.push(row_result);
        }
        result
    }
}
