import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Linking } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Shield, AlertTriangle, BookOpen, ExternalLink, Heart, Phone } from 'lucide-react-native';

export default function InfoScreen() {
  const emergencyCall = () => {
    Linking.openURL('tel:911');
  };

  const openWebsite = (url: string) => {
    Linking.openURL(url);
  };

  const infoSections = [
    {
      icon: Shield,
      title: 'About SkinGuard AI',
      color: '#0066CC',
      content: [
        'SkinGuard AI uses advanced machine learning to analyze skin lesions and moles for potential signs of melanoma and other skin cancers.',
        'Our AI model was trained on thousands of dermatologist-verified images to provide accurate risk assessments.',
        'This tool is designed as a screening aid and should never replace professional medical consultation.',
      ],
    },
    {
      icon: AlertTriangle,
      title: 'ABCDE Signs of Melanoma',
      color: '#F59E0B',
      content: [
        'A - Asymmetry: One half doesn\'t match the other half',
        'B - Border: Edges are irregular, ragged, or blurred',
        'C - Color: Multiple colors or uneven distribution',
        'D - Diameter: Larger than 6mm (size of a pencil eraser)',
        'E - Evolving: Changes in size, shape, color, or texture',
      ],
    },
    {
      icon: Heart,
      title: 'Prevention Tips',
      color: '#10B981',
      content: [
        'Use broad-spectrum sunscreen with SPF 30 or higher',
        'Seek shade during peak UV hours (10 AM - 4 PM)',
        'Wear protective clothing and wide-brimmed hats',
        'Avoid tanning beds and excessive sun exposure',
        'Perform monthly self-examinations',
        'Schedule annual dermatologist check-ups',
      ],
    },
  ];

  const resources = [
    {
      title: 'American Cancer Society',
      url: 'https://www.cancer.org/cancer/melanoma-skin-cancer.html',
      description: 'Comprehensive information about skin cancer',
    },
    {
      title: 'Skin Cancer Foundation',
      url: 'https://www.skincancer.org',
      description: 'Prevention and early detection resources',
    },
    {
      title: 'American Academy of Dermatology',
      url: 'https://www.aad.org',
      description: 'Find a dermatologist near you',
    },
  ];

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Health Information</Text>
          <Text style={styles.subtitle}>Learn about skin cancer detection and prevention</Text>
        </View>

        {/* Emergency Alert */}
        <View style={styles.emergencyAlert}>
          <AlertTriangle size={24} color="#EF4444" strokeWidth={2} />
          <View style={styles.emergencyContent}>
            <Text style={styles.emergencyTitle}>Medical Emergency?</Text>
            <Text style={styles.emergencyText}>
              If you notice rapid changes in a mole or lesion, contact your doctor immediately.
            </Text>
          </View>
          <TouchableOpacity style={styles.emergencyButton} onPress={emergencyCall}>
            <Phone size={16} color="#FFFFFF" strokeWidth={2} />
          </TouchableOpacity>
        </View>

        {/* Information Sections */}
        {infoSections.map((section, index) => (
          <View key={index} style={styles.infoSection}>
            <View style={styles.sectionHeader}>
              <View style={[styles.sectionIcon, { backgroundColor: `${section.color}15` }]}>
                <section.icon size={24} color={section.color} strokeWidth={2} />
              </View>
              <Text style={styles.sectionTitle}>{section.title}</Text>
            </View>
            
            <View style={styles.sectionContent}>
              {section.content.map((item, itemIndex) => (
                <View key={itemIndex} style={styles.contentItem}>
                  <View style={styles.bullet} />
                  <Text style={styles.contentText}>{item}</Text>
                </View>
              ))}
            </View>
          </View>
        ))}

        {/* Disclaimer */}
        <View style={styles.disclaimer}>
          <Text style={styles.disclaimerTitle}>Important Medical Disclaimer</Text>
          <Text style={styles.disclaimerText}>
            SkinGuard AI is not a substitute for professional medical advice, diagnosis, or treatment. 
            Always seek the advice of your physician or other qualified health provider with any questions 
            you may have regarding a medical condition. Never disregard professional medical advice or 
            delay in seeking it because of something you have seen in this app.
          </Text>
        </View>

        {/* Resources */}
        <View style={styles.resourcesSection}>
          <Text style={styles.resourcesTitle}>Additional Resources</Text>
          {resources.map((resource, index) => (
            <TouchableOpacity
              key={index}
              style={styles.resourceCard}
              onPress={() => openWebsite(resource.url)}>
              <View style={styles.resourceContent}>
                <Text style={styles.resourceTitle}>{resource.title}</Text>
                <Text style={styles.resourceDescription}>{resource.description}</Text>
              </View>
              <ExternalLink size={20} color="#0066CC" strokeWidth={2} />
            </TouchableOpacity>
          ))}
        </View>

        {/* App Info */}
        <View style={styles.appInfo}>
          <Text style={styles.appInfoTitle}>SkinGuard AI v1.0</Text>
          <Text style={styles.appInfoText}>
            Developed for educational and screening purposes. 
            Last updated: January 2024
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  header: {
    padding: 24,
    paddingBottom: 16,
  },
  title: {
    fontSize: 28,
    fontFamily: 'Inter-Bold',
    color: '#111827',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
  },
  emergencyAlert: {
    flexDirection: 'row',
    alignItems: 'center',
    margin: 24,
    padding: 16,
    backgroundColor: '#FEF2F2',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#FECACA',
  },
  emergencyContent: {
    flex: 1,
    marginLeft: 12,
  },
  emergencyTitle: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#DC2626',
    marginBottom: 4,
  },
  emergencyText: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#7F1D1D',
  },
  emergencyButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#EF4444',
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 12,
  },
  infoSection: {
    margin: 24,
    marginTop: 0,
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    flex: 1,
  },
  sectionContent: {
    marginLeft: 16,
  },
  contentItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  bullet: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#9CA3AF',
    marginTop: 8,
    marginRight: 12,
  },
  contentText: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#374151',
    lineHeight: 20,
    flex: 1,
  },
  disclaimer: {
    margin: 24,
    marginTop: 0,
    padding: 20,
    backgroundColor: '#FEF3C7',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#FDE68A',
  },
  disclaimerTitle: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#92400E',
    marginBottom: 8,
  },
  disclaimerText: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#78350F',
    lineHeight: 20,
  },
  resourcesSection: {
    margin: 24,
    marginTop: 0,
  },
  resourcesTitle: {
    fontSize: 20,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginBottom: 16,
  },
  resourceCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  resourceContent: {
    flex: 1,
  },
  resourceTitle: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginBottom: 4,
  },
  resourceDescription: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
  },
  appInfo: {
    margin: 24,
    marginTop: 0,
    padding: 16,
    alignItems: 'center',
  },
  appInfoTitle: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginBottom: 4,
  },
  appInfoText: {
    fontSize: 12,
    fontFamily: 'Inter-Regular',
    color: '#9CA3AF',
    textAlign: 'center',
  },
});