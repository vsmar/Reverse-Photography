// Testing script for the photodiode setup

const int photoSensor = A0;

void setup() {
  Serial.begin(9600);
  while (!Serial);
}

void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();
    if (msg.startsWith("Projecting:")) {
      int idx = msg.substring(11).toInt();
      int photoReading = analogRead(photoSensor);
      Serial.print("Reading: ");
      Serial.print(idx);
      Serial.print(", ");
      Serial.println(photoReading);
    }
  }

  delay(10);
}

