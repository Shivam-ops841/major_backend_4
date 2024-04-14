// Import mongoose
const mongoose = require('mongoose');

// Define session schema
const sessionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    required: true,
    ref: 'User' // Reference to the User model
  },
  token: {
    type: String,
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  expiresAt: {
    type: Date,
    required: true
  }
});

// Create and export Session model
const Session = mongoose.model('Session', sessionSchema);

module.exports = Session;
