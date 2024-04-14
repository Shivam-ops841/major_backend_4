// Import mongoose
const mongoose = require('mongoose');

// Define image schema
const imageSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    required: true,
    ref: 'User' // Reference to the User model
  },
  imageName: {
    type: String,
    required: true
  },
  imagePath: {
    type: String,
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

// Create and export Image model
const Image = mongoose.model('Image', imageSchema);

module.exports = Image;