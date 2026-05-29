use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::Point3;

pub struct Sphere {
    center: Point3,
    radius: f64,
    material: Box<dyn Material>,
}

impl Sphere {
    pub fn new(center: Point3, radius: f64, material: impl Material + 'static) -> Self {
        Self {
            center,
            radius,
            material: Box::new(material),
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<(HitRecord, &dyn Material)> {
        let o = ray.origin();
        let d = ray.direction();
        let a = d.length_squared();
        let h = d.dot(self.center - o);
        let c = (self.center - o).length_squared() - self.radius * self.radius;

        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            return None;
        }

        let s = discriminant.sqrt();
        for t in [(h - s) / a, (h + s) / a] {
            if t_min <= t && t <= t_max {
                let p = ray.at(t);
                let normal = (p - self.center).unit_vector();
                return Some((HitRecord::new(t, p, normal, ray), &*self.material));
            }
        }
        None
    }
}
