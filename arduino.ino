#include <Servo.h>

/* -------- Servo Objects -------- */
Servo servo1;   // Rotates chute toward bins
Servo servo2;   // Opens / closes bin lid

/* -------- Pin Definitions -------- */
#define IR_SENSOR 2
#define METAL_SENSOR A5
#define BUZZER 12
#define MOISTURE_SENSOR A0

/* -------- Variables -------- */
int soil = 0;
int fsoil = 0;

void setup()
{
  Serial.begin(9600);

  pinMode(IR_SENSOR, INPUT);
  pinMode(METAL_SENSOR, INPUT);
  pinMode(BUZZER, OUTPUT);

  servo1.attach(9);   // Servo for bin rotation
  servo2.attach(7);   // Servo for lid

  Serial.println("Smart Waste Segregation System Initialized");

  /* Initial Positions */
  servo1.write(70);   // Neutral / Wet bin position
  servo2.write(0);    // Lid closed

  delay(1000);
}

/* -------- Main Loop -------- */
void loop()
{

  int metalState = digitalRead(METAL_SENSOR);

  /* ---------- METAL DETECTION ---------- */
  if (metalState == 0)
  {
    Serial.println("Metal Waste Detected");

    tone(BUZZER, 1000, 1000);
    delay(500);

    /* Rotate chute to METAL bin */
    servo1.write(0);
    Serial.println("Rotating chute to METAL bin");
    delay(5000);

    /* Open lid */
    servo2.write(180);
    Serial.println("Opening lid");
    delay(5000);

    /* Close lid */
    servo2.write(0);
    Serial.println("Closing lid");
    delay(5000);

    /* Return to neutral */
    servo1.write(70);
    delay(1000);

    noTone(BUZZER);
  }

  /* ---------- OBJECT DETECTED BY IR ---------- */
  if (digitalRead(IR_SENSOR) == 0)
  {

    Serial.println("Object detected by IR sensor");

    tone(BUZZER, 1000, 500);
    delay(500);

    fsoil = 0;

    Serial.println("Reading moisture sensor");

    /* Take 3 readings for stability */
    for (int i = 0; i < 3; i++)
    {
      soil = analogRead(MOISTURE_SENSOR);

      soil = constrain(soil, 485, 1023);

      fsoil = map(soil, 485, 1023, 100, 0) + fsoil;

      delay(75);
    }

    fsoil = fsoil / 3;

    Serial.print("Moisture Percentage: ");
    Serial.println(fsoil);

    /* ---------- DRY WASTE ---------- */
    if (fsoil < 40)
    {
      Serial.println("Dry Waste Detected");

      servo1.write(180);   // Rotate to DRY bin
      delay(5000);

      servo2.write(180);   // Open lid
      Serial.println("Opening lid for Dry Waste");
      delay(5000);

      servo2.write(0);     // Close lid
      delay(5000);

      servo1.write(70);    // Return to neutral
      delay(5000);
    }

    /* ---------- WET WASTE ---------- */
    else
    {
      Serial.println("Wet Waste Detected");

      servo1.write(70);   // Wet bin position
      delay(5000);

      servo2.write(180);  // Open lid
      Serial.println("Opening lid for Wet Waste");
      delay(5000);

      servo2.write(0);    // Close lid
      delay(5000);
    }

    noTone(BUZZER);
  }

  delay(200);
}