class Planet {
   
  private PVector location;
  private PVector velocity;
  private PVector acceleration;
  
  static final float maxVelocity = 10; 
  
  Planet() {
     location = new PVector(width/2, height/2);
     setVelocity(new PVector(0, 0));
     acceleration = new PVector(0, 0);
  }
  
  private void setVelocity(PVector velocity) {
    this.velocity = velocity;
    this.velocity.limit(maxVelocity);
  }
  
  private void updateAcceleration() {
    PVector cursor = new PVector(mouseX, mouseY);
    PVector diff = PVector.sub(cursor, location);
    diff.normalize();
    diff.mult(0.05);
    acceleration = diff;
  }
  
  public void update() {
    updateAcceleration();
    velocity.add(acceleration);
    location.add(velocity);
  }
  
  public void display() {
    stroke(0);
    fill(175);
    ellipse(location.x, location.y, 10, 10);
  }
  
}
