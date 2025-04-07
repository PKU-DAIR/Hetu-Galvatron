import React from 'react';
import { Button, Tooltip } from '@mui/material';
import { useLanguage, LANGUAGES } from '../context/LanguageContext';

/**
 * Language toggle button component
 * Switches between Chinese and English languages
 */
function LanguageToggle() {
  const { language, toggleLanguage } = useLanguage();
  
  return (
    <Tooltip title={language === LANGUAGES.ZH ? "Switch to English" : "切换为中文"}>
      <Button 
        onClick={toggleLanguage}
        variant="outlined"
        size="small"
        sx={{ 
          minWidth: 'auto', 
          borderRadius: '50%', 
          width: 40, 
          height: 40,
          fontWeight: 'bold',
          fontSize: '14px',
          color: 'primary.main',
          borderColor: 'primary.main',
          '&:hover': {
            backgroundColor: 'rgba(25, 118, 210, 0.04)',
            borderColor: 'primary.dark',
          }
        }}
      >
        {language === LANGUAGES.ZH ? 'EN' : '中'}
      </Button>
    </Tooltip>
  );
}

export default LanguageToggle; 