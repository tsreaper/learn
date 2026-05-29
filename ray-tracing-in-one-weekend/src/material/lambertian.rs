use crate::hittable::HitRecord;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Color, random_unit_vector};

pub struct Lambertian {
    albedo: Color,
}

impl Lambertian {
    pub fn new(albedo: Color) -> Self {
        Self { albedo }
    }
}

impl Material for Lambertian {
    fn scatter(&self, _: &Ray, hit_record: &HitRecord) -> Option<(Color, Ray)> {
        let mut d = random_unit_vector() + hit_record.normal();
        if d.near_zero() {
            d = hit_record.normal();
        }
        Some((self.albedo, Ray::new(hit_record.p(), d)))
    }
}
