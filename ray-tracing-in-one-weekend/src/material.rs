mod dielectric;
mod lambertian;
mod metal;

pub use dielectric::Dieletric;
pub use lambertian::Lambertian;
pub use metal::Metal;

use crate::hittable::HitRecord;
use crate::ray::Ray;
use crate::vec3::Color;

pub trait Material {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Color, Ray)>;
}
