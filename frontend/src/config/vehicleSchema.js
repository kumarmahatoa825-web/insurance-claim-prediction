/**
 * InsureAI — Canonical Vehicle Insurance Schema (v1.0.0)
 * Synchronized with schemas/vehicle_insurance_v1.json
 * Schema Hash: 76e05d11c94fb38d6732c7ef18b1ef5664d3cacc8ec9497bcbbd5fd190d3d4bb
 */

export const VEHICLE_SCHEMA = {
  schemaName: "vehicle_insurance_v1",
  version: "1.0.0",
  schemaHash: "76e05d11c94fb38d6732c7ef18b1ef5664d3cacc8ec9497bcbbd5fd190d3d4bb",

  areaClusters: [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
    "C22"
],

  makes: [1, 2, 3, 4, 5],

  segments: ["A", "B1", "B2", "C1", "C2", "Utility"],

  models: ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11"],

  fuelTypes: ["CNG", "Diesel", "Petrol"],

  transmissionTypes: ["Automatic", "Manual"],

  steeringTypes: ["Electric", "Manual", "Power"],

  rearBrakesTypes: ["Disc", "Drum"],

  engineTypes: [
    "1.0 SCe",
    "1.2 L K Series Engine",
    "1.2 L K12N Dualjet",
    "1.5 L U2 CRDi",
    "1.5 Turbocharged Revotorq",
    "1.5 Turbocharged Revotron",
    "F8D Petrol Engine",
    "G12B",
    "K Series Dual jet",
    "K10C",
    "i-DTEC"
],

  binaryFields: [
    "is_esc",
    "is_adjustable_steering",
    "is_tpms",
    "is_parking_sensors",
    "is_parking_camera",
    "is_front_fog_lights",
    "is_rear_window_wiper",
    "is_rear_window_washer",
    "is_rear_window_defogger",
    "is_brake_assist",
    "is_power_door_locks",
    "is_central_locking",
    "is_power_steering",
    "is_driver_seat_height_adjustable",
    "is_day_night_rear_view_mirror",
    "is_ecw",
    "is_speed_alert"
],

  numericBounds: {
    policy_tenure: { min: 0.0, max: 2.0, step: 0.01, label: "Policy Tenure (Dataset Normalized, 0–1.4)" },
    age_of_car: { min: 0.0, max: 1.0, step: 0.01, label: "Age of Car (Dataset Normalized, 0–1.0)" },
    age_of_policyholder: { min: 0.0, max: 1.0, step: 0.01, label: "Age of Policyholder (Dataset Normalized, 0–1.0)" },
    population_density: { min: 0, max: 150000, step: 10, label: "Population Density (people/km²)" },
    airbags: { min: 1, max: 10, step: 1, label: "Airbags Count" },
    displacement: { min: 500, max: 2500, step: 1, label: "Engine Displacement (cc)" },
    cylinder: { min: 2, max: 8, step: 1, label: "Cylinders" },
    gear_box: { min: 4, max: 8, step: 1, label: "Gear Box Speeds" },
    turning_radius: { min: 3.0, max: 10.0, step: 0.1, label: "Turning Radius (m)" },
    length: { min: 2000, max: 6000, step: 1, label: "Length (mm)" },
    width: { min: 1000, max: 3000, step: 1, label: "Width (mm)" },
    height: { min: 1000, max: 3000, step: 1, label: "Height (mm)" },
    gross_weight: { min: 500, max: 4000, step: 1, label: "Gross Weight (kg)" },
    ncap_rating: { min: 0, max: 5, step: 1, label: "NCAP Safety Rating (0–5)" }
  }
};

export const DEFAULT_VEHICLE_RECORD = {
  policy_id: "POL-2024-001",
  policy_tenure: 0.5,
  age_of_car: 0.1,
  age_of_policyholder: 0.45,
  area_cluster: "C1",
  population_density: 12000,
  make: 1,
  segment: "A",
  model: "M1",
  fuel_type: "Petrol",
  max_torque: "100Nm@4000rpm",
  max_power: "80bhp@5500rpm",
  engine_type: "1.2 L K Series Engine",
  airbags: 2,
  is_esc: "No",
  is_adjustable_steering: "Yes",
  is_tpms: "No",
  is_parking_sensors: "Yes",
  is_parking_camera: "No",
  rear_brakes_type: "Drum",
  displacement: 1200,
  cylinder: 4,
  transmission_type: "Manual",
  gear_box: 5,
  steering_type: "Power",
  turning_radius: 5.0,
  length: 3900,
  width: 1700,
  height: 1550,
  gross_weight: 1400,
  is_front_fog_lights: "No",
  is_rear_window_wiper: "No",
  is_rear_window_washer: "No",
  is_rear_window_defogger: "No",
  is_brake_assist: "No",
  is_power_door_locks: "Yes",
  is_central_locking: "Yes",
  is_power_steering: "Yes",
  is_driver_seat_height_adjustable: "No",
  is_day_night_rear_view_mirror: "No",
  is_ecw: "Yes",
  is_speed_alert: "Yes",
  ncap_rating: 2,
  store_prediction: true,
};
