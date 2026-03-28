// MQ135 Sensor Pin
int mq135 = A0;

// Dummy sensors (replace later if needed)
float temperature = 25.0;
float humidity = 60.0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int gasValue = analogRead(mq135);

  // Convert to dummy pollution values
  float PM25 = gasValue * 0.5;
  float PM10 = gasValue * 0.8;
  float CO = gasValue * 0.02;
  float NO2 = gasValue * 0.03;
  float SO2 = gasValue * 0.01;
  float O3 = gasValue * 0.015;

  // Send data in CSV format
  Serial.print(PM25); Serial.print(",");
  Serial.print(PM10); Serial.print(",");
  Serial.print(NO2); Serial.print(",");
  Serial.print(CO); Serial.print(",");
  Serial.print(SO2); Serial.print(",");
  Serial.print(O3); Serial.print(",");
  Serial.print(temperature); Serial.print(",");
  Serial.println(humidity);

  delay(2000);
}