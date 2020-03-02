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
    setVelocity(PVector.add(velocity, acceleration));
    location.add(velocity);
    checkEdges();
  }
  
  public void display() {
    stroke(0);
    fill(175);
    ellipse(location.x, location.y, 10, 10);
  }
  
  private void checkEdges() {
    if (location.x > width)
      location.x = 0;
    else if (location.x < 0)
      location.x = width;
    
    if (location.y > height)
      location.y = 0;
    else if (location.y < 0)
      location.y = height;
  }

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
    setVelocity(PVector.add(velocity, acceleration));
    location.add(velocity);
    checkEdges();
  }

  public void display() {
    stroke(0);
    fill(175);
    ellipse(location.x, location.y, 10, 10);
  }

  private void checkEdges() {
    if (location.x > width)
      location.x = 0;
    else if (location.x < 0)
      location.x = width;

    if (location.y > height)
      location.y = 0;
    else if (location.y < 0)
      location.y = height;
  }

}
