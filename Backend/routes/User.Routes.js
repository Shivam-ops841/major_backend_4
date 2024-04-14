const express=require('express');
const route=express.Router();
const {getProfile,updateProfile}=require('../controllers/User.Controller')

route.get('/profile/:id',getProfile)
route.put('/profile',updateProfile)

module.exports=route