const User=require('../models/User.Model' )
const jwt=require('jsonwebtoken')
const bcrypt=require('bcrypt')

exports.register = async (req, res) => {
    try {
        // Extract user data from request body
        const { username, email, password } = req.body;

        // Check if username or email already exists
        let existingUser = await User.findOne({ $or: [{ username }, { email }] });
        if (existingUser) {
            return res.status(400).json({ message: 'Username or email already exists' });
        }

        // Hash the password
        const hashedPassword = await bcrypt.hash(password, 10);

        // Create a new user in the database
        const newUser = new User({
            username,
            email,
            password: hashedPassword
        });
        await newUser.save();

        // Generate JWT token
        const token = jwt.sign({ userId: newUser._id }, process.env.JWT_SECRET, { expiresIn: '1d' });

        // Send JWT token as response
        res.status(201).json({ token });
    } catch (error) {
        console.error('Error in register controller:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
};

// Login Controller
exports.login = async (req, res) => {
    try {
        // Extract user credentials from request body
        const { email, password } = req.body;

        // Find user by username in the database
        const user = await User.findOne({ email });
        if (!user) {
            return res.status(401).json({ message: 'Invalid username or password' });
        }

        // Compare hashed password with provided password
        const passwordMatch = await bcrypt.compare(password, user.password);
        if (!passwordMatch) {
            return res.status(401).json({ message: 'Invalid username or password' });
        }

        // Generate JWT token
        const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET, { expiresIn: '1d' });

        // Send JWT token as response
        res.json({ token });
    } catch (error) {
        console.error('Error in login controller:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
};

//Logout Controller
exports.logout = async (req, res) => {
    // To be handled in frontend
    res.status(200).json({ message: 'Logout successful' });
};