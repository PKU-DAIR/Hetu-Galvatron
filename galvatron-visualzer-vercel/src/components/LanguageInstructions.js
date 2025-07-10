import React from 'react';
import { Paper, Typography, Box, Grow } from '@mui/material';
import { useLanguage, LANGUAGES } from '../context/LanguageContext';

/**
 * Brief instructions component for language toggle feature
 * Shows a temporary notification about the language toggle
 */
function LanguageInstructions() {
  const { language } = useLanguage();
  const [visible, setVisible] = React.useState(true);
  
  // Hide notification after 10 seconds
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(false);
    }, 10000);
    
    return () => clearTimeout(timer);
  }, []);
  
  if (!visible) return null;
  
  return (
    <Grow in={visible}>
      <Paper 
        elevation={2}
        sx={{
          position: 'fixed',
          bottom: 20,
          right: 20,
          maxWidth: 300,
          p: 2,
          backgroundColor: 'rgba(25, 118, 210, 0.9)',
          color: 'white',
          zIndex: 1000,
          borderRadius: 2
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 1 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
            {language === LANGUAGES.ZH ? '新功能：语言切换' : 'New Feature: Language Toggle'}
          </Typography>
        </Box>
        
        <Typography variant="body2" sx={{ mb: 1 }}>
          {language === LANGUAGES.ZH 
            ? '您现在可以使用右上角的语言切换按钮在中文和英文之间切换界面语言。' 
            : 'You can now switch between Chinese and English using the language toggle button in the top right corner.'}
        </Typography>
        
        <Typography variant="caption" sx={{ opacity: 0.8 }}>
          {language === LANGUAGES.ZH ? '此提示将在几秒后消失' : 'This notification will disappear in a few seconds'}
        </Typography>
      </Paper>
    </Grow>
  );
}

export default LanguageInstructions; 