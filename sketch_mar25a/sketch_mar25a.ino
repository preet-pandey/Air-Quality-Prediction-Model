#include <Wire.h> 
#include <LiquidCrystal_I2C.h>
#include <DHT.h>

/**
 * AeroGuard - Full System (Sensors + LCD + Backend)
 * 
 * Connections:
 * LCD I2C: VCC->5V, GND->GND, SDA->A4, SCL->A5
 * MQ135:   VCC->5V, GND->GND, AO->A0
 * DHT11:   VCC->5V, GND->GND, DATA->D2
 */

// LCD Setup (Address 0x27 or 0x3F)
LiquidCrystal_I2C lcd(0x27, 16, 2); 

// Sensor Setup
#define MQ135_PIN A0
#define DHT_PIN 2
#define DHT_TYPE DHT11
DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(9600);
  
  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.print("AeroGuard v1.0");
  lcd.setCursor(0, 1);
  lcd.print("Starting up...");
  
  // Initialize DHT
  dht.begin();
  
  delay(2000);
}

void loop() {
  // 1. Read Sensors
  int gas = analogRead(MQ135_PIN);
  float hum = dht.readHumidity();
  float temp = dht.readTemperature();

  // Handle sensor errors
  if (isnan(hum) || isnan(temp)) {
    hum = 0.0;
    temp = 0.0;
  }

  // 2. Update LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Gas: "); lcd.print(gas);
  
  lcd.setCursor(0, 1);
  lcd.print("T:"); lcd.print((int)temp); lcd.print("C ");
  lcd.print("H:"); lcd.print((int)hum); lcd.print("%");

  // 3. Send to Backend (Format: GAS,TEMP,HUMIDITY)
  Serial.print(gas);
  Serial.print(",");
  Serial.print(temp);
  Serial.print(",");
  Serial.println(hum);

  delay(2000); // 2-second interval for stability
}
