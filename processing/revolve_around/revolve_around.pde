Planet planet;

void setup() {
   background(255);
   size(500, 500);
   planet = new Planet();
}

void draw() {
   planet.update();
   planet.display();
}
