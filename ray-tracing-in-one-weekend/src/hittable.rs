use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

pub trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord<'_>>;
}

pub struct HitRecord<'a> {
    p: Point3,
    normal: Vec3,
    t: f64,
    material: &'a dyn Material,
    front_face: bool,
}

impl<'a> HitRecord<'a> {
    pub fn new(
        p: Point3,
        outward_normal: Vec3,
        t: f64,
        material: &'a dyn Material,
        ray: &Ray,
    ) -> Self {
        let (normal, front_face) = if outward_normal.dot(ray.direction()) > 0.0 {
            (-outward_normal, false)
        } else {
            (outward_normal, true)
        };
        Self {
            p,
            normal,
            t,
            material,
            front_face,
        }
    }

    pub fn p(&self) -> Point3 {
        self.p
    }

    pub fn normal(&self) -> Vec3 {
        self.normal
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn material(&self) -> &'a dyn Material {
        self.material
    }
}

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
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord<'_>> {
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
                return Some(HitRecord::new(p, normal, t, &*self.material, ray));
            }
        }
        None
    }
}

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
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord<'_>> {
        let mut current_closest = t_max;
        let mut result = None;
        for obj in self.objects.iter() {
            match obj.hit(ray, t_min, current_closest) {
                Some(r) => {
                    current_closest = r.t();
                    result = Some(r);
                }
                None => {}
            }
        }
        result
    }
}
