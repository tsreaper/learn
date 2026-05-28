mod vec3;
mod ray;

use vec3::{Vec3, Color, Point3};
use ray::Ray;

fn write_color(color: &Color) {
    let ir = (255.999 * color.x()) as i32;
    let ig = (255.999 * color.y()) as i32;
    let ib = (255.999 * color.z()) as i32;
    println!("{} {} {}", ir, ig, ib);
}

fn ray_color(ray: &Ray) -> Color {
    let unit_direction = ray.direction().unit_vector();
    let a = 0.5 * (unit_direction.y() + 1.0);
    (1.0 - a) * Color::new(1.0, 1.0, 1.0) + a * Color::new(0.5, 0.7, 1.0)
}

fn main() {
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 400.0;
    let image_height = f64::max(1.0, image_width / aspect_ratio);

    let focal_length = 1.0;
    let viewport_height = 2.0;
    let viewport_width = viewport_height * image_width / image_height;
    let camera_center = Point3::new(0.0, 0.0, 0.0);

    let viewport_u = Vec3::new(viewport_width, 0.0, 0.0);
    let viewport_v = Vec3::new(0.0, -viewport_height, 0.0);

    let pixel_delta_u = viewport_u / image_width;
    let pixel_delta_v = viewport_v / image_height;

    let viewport_upper_left = camera_center - Vec3::new(0.0, 0.0, focal_length) - viewport_u / 2.0 - viewport_v / 2.0;
    let pixel00_loc = viewport_upper_left + 0.5 * pixel_delta_u + 0.5 * pixel_delta_v;

    println!("P3\n{} {}\n255", image_width as i32, image_height as i32);

    for i in 0..image_height as u32 {
        eprint!("\rScanlines remaining: {}    ", image_height as u32 - i);
        for j in 0..image_width as u32 {
            let pixel_center = pixel00_loc + (j as f64 * pixel_delta_u) + (i as f64 * pixel_delta_v);
            let ray_direction = pixel_center - camera_center;
            let ray = Ray::new(pixel_center, ray_direction);

            let pixel_color = ray_color(&ray);
            write_color(&pixel_color);
        }
    }

    eprintln!("\rDone.                         ",);
}
