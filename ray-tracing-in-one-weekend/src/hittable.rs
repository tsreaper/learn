mod hittable_list;
mod sphere;

pub use hittable_list::HittableList;
pub use sphere::Sphere;

use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

pub trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<(HitRecord, &dyn Material)>;
}

pub struct HitRecord {
    t: f64,
    p: Point3,
    normal: Vec3,
    front_face: bool,
}

impl HitRecord {
    pub fn new(t: f64, p: Point3, outward_normal: Vec3, ray: &Ray) -> Self {
        let (normal, front_face) = if outward_normal.dot(ray.direction()) > 0.0 {
            (-outward_normal, false)
        } else {
            (outward_normal, true)
        };
        Self {
            t,
            p,
            normal,
            front_face,
        }
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn p(&self) -> Point3 {
        self.p
    }

    pub fn normal(&self) -> Vec3 {
        self.normal
    }

    pub fn front_face(&self) -> bool {
        self.front_face
    }
}
