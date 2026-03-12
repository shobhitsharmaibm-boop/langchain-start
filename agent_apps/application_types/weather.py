from typing import TypedDict, List

class Location(TypedDict):
    lat: float
    lng: float

class Geometry(TypedDict):
    location: Location

class Result(TypedDict):
    geometry: Geometry

class GeocodeResponse(TypedDict):
    results: List[Result]
    status: str


class TimeZone(TypedDict):
    id: str

class WeatherDescription(TypedDict):
    text: str
    languageCode: str

class WeatherCondition(TypedDict):
    iconBaseUri: str
    description: WeatherDescription
    type: str

class Temperature(TypedDict):
    degrees: float
    unit: str

class PrecipitationProbability(TypedDict):
    percent: int
    type: str

class QPF(TypedDict):
    quantity: float
    unit: str

class Precipitation(TypedDict):
    probability: PrecipitationProbability
    qpf: QPF

class AirPressure(TypedDict):
    meanSeaLevelMillibars: float

class WindDirection(TypedDict):
    degrees: int
    cardinal: str

class WindValue(TypedDict):
    value: float
    unit: str

class Wind(TypedDict):
    direction: WindDirection
    speed: WindValue
    gust: WindValue

class Visibility(TypedDict):
    distance: float
    unit: str

class CurrentConditionsHistory(TypedDict):
    temperatureChange: Temperature
    maxTemperature: Temperature
    minTemperature: Temperature
    qpf: QPF

class Weather(TypedDict):
    currentTime: str
    timeZone: TimeZone
    isDaytime: bool
    weatherCondition: WeatherCondition
    temperature: Temperature
    feelsLikeTemperature: Temperature
    dewPoint: Temperature
    heatIndex: Temperature
    windChill: Temperature
    relativeHumidity: int
    uvIndex: int
    precipitation: Precipitation
    thunderstormProbability: int
    airPressure: AirPressure
    wind: Wind
    visibility: Visibility
    cloudCover: int
    currentConditionsHistory: CurrentConditionsHistory