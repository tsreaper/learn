use crate::hittable::HitRecord;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Color, random_unit_vector};

pub struct Metal {
    albedo: Color,
    fuzz: f64,
}

impl Metal {
    pub fn new(albedo: Color, fuzz: f64) -> Self {
        Self { albedo, fuzz }
    }
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Color, Ray)> {
        let d = ray.direction().reflect(hit_record.normal()).unit_vector()
            + (self.fuzz * random_unit_vector());
        if d.dot(hit_record.normal()) > 0.0 {
            Some((self.albedo, Ray::new(hit_record.p(), d)))
        } else {
            None
        }
    }
}
