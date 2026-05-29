use crate::hittable::HitRecord;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::Color;

pub struct Dieletric {
    refraction_index: f64,
}

impl Dieletric {
    pub fn new(refraction_index: f64) -> Self {
        Self { refraction_index }
    }
}

fn reflectance(cos_theta: f64, refraction_index: f64) -> f64 {
    let mut r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cos_theta).powi(5)
}

impl Material for Dieletric {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Color, Ray)> {
        let ri = if hit_record.front_face() {
            1.0 / self.refraction_index
        } else {
            self.refraction_index
        };

        let unit_direction = ray.direction().unit_vector();
        let cos_theta = -unit_direction.dot(hit_record.normal()).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let d = if ri * sin_theta > 1.0 || reflectance(cos_theta, ri) > rand::random_range(0.0..1.0)
        {
            unit_direction.reflect(hit_record.normal())
        } else {
            unit_direction.refract(hit_record.normal(), ri)
        };
        Some((Color::new(1.0, 1.0, 1.0), Ray::new(hit_record.p(), d)))
    }
}
