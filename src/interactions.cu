#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

#include <glm/gtx/norm.hpp>

// world space distance a scattered ray's origin is pushed off the surface
#define SCATTER_EPSILON 1e-3f

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng,
    bool outside)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (m.hasRefractive > 0.0f)
    {
        glm::vec3 I = glm::normalize(pathSegment.ray.direction);
        float eta = outside ? 1.0f / m.indexOfRefraction : m.indexOfRefraction;
        float cosThetaI = glm::dot(-I, normal);

        // Since glm::refract can produce NaN with tir,
        // we compute k using Snell's law, and check if it is valid
        // before refracting.
        float dotNI = glm::dot(normal, I);
        float k = 1.0f - eta * eta * (1.0f - dotNI * dotNI);
        bool tir = k < 0.0f;

        // Schlick approximation
        float r0 = (1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction);
        r0 *= r0;

        // Since we want the cosine on the less dense side (air), we
        // check here if we are entering or exiting and use the proper cosine value.
        float cosSchlick = outside ? cosThetaI : glm::sqrt(glm::max(k, 0.0f));
        // we take into account a negative k value if tir is true.
        float fresnel = tir ? 1.0f : r0 + (1.0f - r0) * powf(1.0f - cosSchlick, 5.0f);

        if (tir || u01(rng) < fresnel)
        {
            pathSegment.ray.direction = glm::reflect(I, normal);
        }
        else
        {
            // No tir, so we won't get NaN
            // finish up the rest of Snell's law to get the refracted direction
            glm::vec3 refracted = eta * I - (eta * dotNI + glm::sqrt(k)) * normal;
            pathSegment.ray.direction = glm::normalize(refracted);
        }
        pathSegment.color *= m.color;
    }
    else if (m.hasReflective > 0.0f)
    {
        pathSegment.ray.direction = glm::reflect(glm::normalize(pathSegment.ray.direction), normal);
        pathSegment.color *= m.color;
    }
    else
    {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
    }

    // We offset by a positive amount when bouncing off, and a negative amount when refracting.
    // This ensures the ray doesn't continuously bounce off a surface when it reaches it.
    pathSegment.ray.origin = intersect + normal * (glm::dot(pathSegment.ray.direction, normal) > 0.0f ? SCATTER_EPSILON : -SCATTER_EPSILON);
    --pathSegment.remainingBounces;
}
