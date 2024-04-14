const User=require('../models/User.Model')

// User Profile Controller
exports.getProfile = async (req, res) => {
   try {
     try {
        const id=req.params.id

        const user=await User.findOne({_id:id})

        res.status(200).json(user)
         } catch (error) {
         console.error('Error in getProfile controller:', error);
         res.status(500).json({ message: 'Internal server error' });
     }
   } catch (error) {
    
   }
};

// Update User Profile Controller
exports.updateProfile = async (req, res) => {
    try {
        // Extract user ID from JWT token
        const userId = req.user.userId;

        // Extract updated profile data from request body
        const { username, email } = req.body;

        // Update user profile information in the database
        await User.findByIdAndUpdate(userId, { username, email });

        // Send success response
        res.status(200).json({ message: 'Profile updated successfully' });
    } catch (error) {
        console.error('Error in updateProfile controller:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
};