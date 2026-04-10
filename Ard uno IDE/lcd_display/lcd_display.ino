#include <Wire.h> 
#include <LiquidCrystal_I2C.h>

/**
 * Arduino Uno & I2C LCD 16x2 Setup
 * 
 * Connections:
 * VCC -> 5V
 * GND -> GND
 * SDA -> A4
 * SCL -> A5
 * 
 * Note: If 0x27 doesn't work, try 0x3F.
 */

// Initialize the LCD (Address, Columns, Rows)
LiquidCrystal_I2C lcd(0x27, 16, 2); 

void setup() {
  // Initialize LCD
  lcd.init();
  
  // Turn on the backlight
  lcd.backlight();
  
  // Clear any existing text
  lcd.clear();
  
  // Set cursor to First Line (Column 0, Row 0)
  lcd.setCursor(0, 0);
  lcd.print("Hello Preet!");
  
  // Set cursor to Second Line (Column 0, Row 1)
  lcd.setCursor(0, 1);
  lcd.print("LCD Connected");
}

void loop() {
  // Static display - No loop actions needed
}
