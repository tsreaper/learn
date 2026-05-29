use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::ray::Ray;

pub struct HittableList {
    objects: Vec<Box<dyn Hittable>>,
}

impl HittableList {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    pub fn add<T: Hittable + 'static>(&mut self, object: T) {
        self.objects.push(Box::new(object));
    }
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<(HitRecord, &dyn Material)> {
        let mut current_closest = t_max;
        let mut result = None;
        for obj in self.objects.iter() {
            match obj.hit(ray, t_min, current_closest) {
                Some((record, material)) => {
                    current_closest = record.t();
                    result = Some((record, material));
                }
                None => {}
            }
        }
        result
    }
}
