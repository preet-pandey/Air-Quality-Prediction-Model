#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// Sensor Pins
#define MQ135_PIN A0
#define DHT_PIN 2      // Connected to Digital Pin 2
#define DHT_TYPE DHT11 // Change to DHT22 if using DHT22

DHT dht(DHT_PIN, DHT_TYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2); // Change to 0x3F if 0x27 doesn't work

void setup() {
  Serial.begin(9600);
  dht.begin();
  
  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.print("AeroGuard Init..");
}

void loop() {
  // 1. Read Sensors
  int gasValue = analogRead(MQ135_PIN);
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  if (isnan(humidity) || isnan(temperature)) {
    humidity = 60.0;
    temperature = 25.0;
  }

  // 2. Send Data to Backend (GAS,TEMP,HUMIDITY)
  Serial.print(gasValue);
  Serial.print(",");
  Serial.print(temperature);
  Serial.print(",");
  Serial.println(humidity);

  // 3. Read predicted AQI from Backend (if available)
  if (Serial.available() > 0) {
    String aqiStr = Serial.readStringUntil('\n');
    aqiStr.trim();
    if (aqiStr.length() > 0) {
      float aqi = aqiStr.toFloat();
      
      // 4. Update LCD
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("AQI: ");
      lcd.print(aqi);
      
      lcd.setCursor(0, 1);
      if (aqi <= 50) lcd.print("Status: Good");
      else if (aqi <= 100) lcd.print("Status: Moderate");
      else if (aqi <= 150) lcd.print("Status: Poor");
      else if (aqi <= 200) lcd.print("Status: Unhealthy");
      else lcd.print("Status: HAZARD!");
    }
  }

  delay(2000); 
}