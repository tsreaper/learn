use crate::hittable::HitRecord;
use crate::ray::Ray;
use crate::vec3::{Color, random_unit_vector};

pub trait Material {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Color, Ray)>;
}

pub struct Lambertian {
    albedo: Color,
}

impl Lambertian {
    pub fn new(albedo: Color) -> Self {
        Self { albedo }
    }
}

impl Material for Lambertian {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Color, Ray)> {
        let mut d = random_unit_vector() + hit_record.normal();
        if d.near_zero() {
            d = hit_record.normal();
        }
        Some((self.albedo, Ray::new(hit_record.p(), d)))
    }
}

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
