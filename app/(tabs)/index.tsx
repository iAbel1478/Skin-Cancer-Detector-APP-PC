import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Dimensions, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Camera, Upload, Shield, TriangleAlert as AlertTriangle, CircleCheck as CheckCircle, TrendingUp, Monitor, Smartphone } from 'lucide-react-native';
import { useRouter } from 'expo-router';
import { useScans } from '@/hooks/useScans';

const { width } = Dimensions.get('window');
const isWeb = Platform.OS === 'web';
const isLargeScreen = width > 768;

export default function HomeScreen() {
  const router = useRouter();
  const { scans } = useScans();

  const quickActions = [
    {
      id: 1,
      title: 'Take Photo',
      subtitle: isWeb ? 'Use webcam or upload' : 'Scan with camera',
      icon: Camera,
      color: '#0066CC',
      bgColor: '#EBF4FF',
      onPress: () => router.push('/camera'),
    },
    {
      id: 2,
      title: 'Upload Image',
      subtitle: 'Select from device',
      icon: Upload,
      color: '#00A693',
      bgColor: '#E6F7F5',
      onPress: () => router.push('/upload'),
    },
  ];

  const stats = [
    { 
      label: 'Scans Completed', 
      value: scans.filter(s => s.result !== 'analyzing').length.toString(), 
      icon: Shield, 
      color: '#0066CC' 
    },
    { 
      label: 'Looks Good', 
      value: scans.filter(s => s.result === 'benign').length.toString(), 
      icon: CheckCircle, 
      color: '#10B981' 
    },
    { 
      label: 'Needs Attention', 
      value: scans.filter(s => s.result === 'concerning').length.toString(), 
      icon: AlertTriangle, 
      color: '#F59E0B' 
    },
    { 
      label: 'Accuracy Rate', 
      value: '94%', 
      icon: TrendingUp, 
      color: '#8B5CF6' 
    },
  ];

  const platformFeatures = [
    {
      icon: isWeb ? Monitor : Smartphone,
      title: isWeb ? 'Desktop Analysis' : 'Mobile Scanning',
      description: isWeb 
        ? 'High-resolution analysis with detailed reporting and export capabilities'
        : 'Instant scanning with real-time AI analysis and cloud sync',
    }
  ];

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView 
        showsVerticalScrollIndicator={false}
        contentContainerStyle={isLargeScreen ? styles.largeScreenContainer : undefined}
      >
        {/* Header */}
        <View style={[styles.header, isLargeScreen && styles.headerLarge]}>
          <Text style={[styles.title, isLargeScreen && styles.titleLarge]}>SkinGuard AI</Text>
          <Text style={[styles.subtitle, isLargeScreen && styles.subtitleLarge]}>
            Advanced Skin Cancer Detection - {isWeb ? 'Desktop' : 'Mobile'} Edition
          </Text>
        </View>

        {/* Hero Section */}
        <LinearGradient
          colors={['#0066CC', '#0052A3']}
          style={[styles.heroCard, isLargeScreen && styles.heroCardLarge]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}>
          <Text style={[styles.heroTitle, isLargeScreen && styles.heroTitleLarge]}>
            Early Detection Saves Lives
          </Text>
          <Text style={[styles.heroSubtitle, isLargeScreen && styles.heroSubtitleLarge]}>
            Use AI-powered analysis to check moles and skin lesions for potential signs of skin cancer. 
            {isWeb ? ' Optimized for desktop with enhanced analysis capabilities.' : ' Instant mobile scanning with cloud sync.'}
          </Text>
        </LinearGradient>

        <View style={isLargeScreen ? styles.contentGrid : styles.contentStack}>
          {/* Quick Actions */}
          <View style={[styles.section, isLargeScreen && styles.sectionLarge]}>
            <Text style={[styles.sectionTitle, isLargeScreen && styles.sectionTitleLarge]}>
              Quick Scan
            </Text>
            <View style={[
              styles.actionsContainer, 
              isLargeScreen && styles.actionsContainerLarge
            ]}>
              {quickActions.map((action) => (
                <TouchableOpacity
                  key={action.id}
                  style={[
                    styles.actionCard, 
                    { backgroundColor: action.bgColor },
                    isLargeScreen && styles.actionCardLarge
                  ]}
                  onPress={action.onPress}>
                  <View style={[styles.actionIcon, { backgroundColor: action.color }]}>
                    <action.icon size={isLargeScreen ? 32 : 24} color="#FFFFFF" strokeWidth={2} />
                  </View>
                  <Text style={[styles.actionTitle, isLargeScreen && styles.actionTitleLarge]}>
                    {action.title}
                  </Text>
                  <Text style={[styles.actionSubtitle, isLargeScreen && styles.actionSubtitleLarge]}>
                    {action.subtitle}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Stats */}
          <View style={[styles.section, isLargeScreen && styles.sectionLarge]}>
            <Text style={[styles.sectionTitle, isLargeScreen && styles.sectionTitleLarge]}>
              Your Health Stats
            </Text>
            <View style={[
              styles.statsContainer,
              isLargeScreen && styles.statsContainerLarge
            ]}>
              {stats.map((stat, index) => (
                <View key={index} style={[
                  styles.statCard,
                  isLargeScreen && styles.statCardLarge
                ]}>
                  <View style={[styles.statIcon, { backgroundColor: `${stat.color}15` }]}>
                    <stat.icon size={isLargeScreen ? 24 : 20} color={stat.color} strokeWidth={2} />
                  </View>
                  <Text style={[styles.statValue, isLargeScreen && styles.statValueLarge]}>
                    {stat.value}
                  </Text>
                  <Text style={[styles.statLabel, isLargeScreen && styles.statLabelLarge]}>
                    {stat.label}
                  </Text>
                </View>
              ))}
            </View>
          </View>
        </View>

        {/* Platform Features */}
        <View style={[styles.section, isLargeScreen && styles.sectionLarge]}>
          <Text style={[styles.sectionTitle, isLargeScreen && styles.sectionTitleLarge]}>
            Platform Features
          </Text>
          {platformFeatures.map((feature, index) => (
            <View key={index} style={[styles.featureCard, isLargeScreen && styles.featureCardLarge]}>
              <View style={styles.featureHeader}>
                <feature.icon size={isLargeScreen ? 28 : 24} color="#0066CC" strokeWidth={2} />
                <Text style={[styles.featureTitle, isLargeScreen && styles.featureTitleLarge]}>
                  {feature.title}
                </Text>
              </View>
              <Text style={[styles.featureContent, isLargeScreen && styles.featureContentLarge]}>
                {feature.description}
              </Text>
            </View>
          ))}
        </View>

        {/* Health Tips */}
        <View style={[styles.section, isLargeScreen && styles.sectionLarge]}>
          <Text style={[styles.sectionTitle, isLargeScreen && styles.sectionTitleLarge]}>
            Health Tips
          </Text>
          <View style={[styles.tipCard, isLargeScreen && styles.tipCardLarge]}>
            <View style={styles.tipHeader}>
              <Shield size={isLargeScreen ? 24 : 20} color="#0066CC" strokeWidth={2} />
              <Text style={[styles.tipTitle, isLargeScreen && styles.tipTitleLarge]}>
                Daily Skin Check
              </Text>
            </View>
            <Text style={[styles.tipContent, isLargeScreen && styles.tipContentLarge]}>
              Examine your skin regularly and look for any changes in moles, freckles, or other marks. 
              Use SkinGuard AI to document and analyze suspicious areas. 
              {isWeb ? ' Take advantage of the larger screen for detailed analysis and comparison.' : ' Use the mobile app for quick daily checks.'}
            </Text>
          </View>
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
  largeScreenContainer: {
    maxWidth: 1200,
    alignSelf: 'center',
    width: '100%',
  },
  header: {
    padding: 24,
    paddingBottom: 16,
  },
  headerLarge: {
    padding: 40,
    paddingBottom: 24,
    alignItems: 'center',
  },
  title: {
    fontSize: 32,
    fontFamily: 'Inter-Bold',
    color: '#111827',
    marginBottom: 4,
  },
  titleLarge: {
    fontSize: 48,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
  },
  subtitleLarge: {
    fontSize: 20,
    textAlign: 'center',
    maxWidth: 600,
  },
  heroCard: {
    margin: 24,
    marginTop: 16,
    padding: 24,
    borderRadius: 16,
  },
  heroCardLarge: {
    margin: 40,
    marginTop: 24,
    padding: 40,
    borderRadius: 24,
  },
  heroTitle: {
    fontSize: 24,
    fontFamily: 'Inter-Bold',
    color: '#FFFFFF',
    marginBottom: 8,
  },
  heroTitleLarge: {
    fontSize: 36,
    textAlign: 'center',
    marginBottom: 16,
  },
  heroSubtitle: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#E5E7EB',
    lineHeight: 24,
  },
  heroSubtitleLarge: {
    fontSize: 18,
    lineHeight: 28,
    textAlign: 'center',
  },
  contentGrid: {
    flexDirection: 'row',
    gap: 40,
    paddingHorizontal: 40,
  },
  contentStack: {
    flexDirection: 'column',
  },
  section: {
    padding: 24,
    paddingTop: 16,
  },
  sectionLarge: {
    flex: 1,
    padding: 0,
    paddingTop: 0,
  },
  sectionTitle: {
    fontSize: 20,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginBottom: 16,
  },
  sectionTitleLarge: {
    fontSize: 28,
    marginBottom: 24,
  },
  actionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 16,
  },
  actionsContainerLarge: {
    flexDirection: 'column',
    gap: 20,
  },
  actionCard: {
    flex: 1,
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
  },
  actionCardLarge: {
    padding: 32,
    borderRadius: 20,
    flexDirection: 'row',
    alignItems: 'center',
    textAlign: 'left',
  },
  actionIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  actionTitle: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginBottom: 4,
  },
  actionTitleLarge: {
    fontSize: 20,
    marginLeft: 20,
    marginBottom: 8,
  },
  actionSubtitle: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    textAlign: 'center',
  },
  actionSubtitleLarge: {
    fontSize: 16,
    marginLeft: 20,
    textAlign: 'left',
  },
  statsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    gap: 12,
  },
  statsContainerLarge: {
    gap: 20,
  },
  statCard: {
    width: (width - 60) / 2,
    backgroundColor: '#FFFFFF',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  statCardLarge: {
    width: '100%',
    padding: 24,
    borderRadius: 16,
    flexDirection: 'row',
    alignItems: 'center',
  },
  statIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  statValue: {
    fontSize: 24,
    fontFamily: 'Inter-Bold',
    color: '#111827',
    marginBottom: 4,
  },
  statValueLarge: {
    fontSize: 32,
    marginLeft: 16,
    marginBottom: 0,
  },
  statLabel: {
    fontSize: 12,
    fontFamily: 'Inter-Medium',
    color: '#6B7280',
    textAlign: 'center',
  },
  statLabelLarge: {
    fontSize: 16,
    marginLeft: 16,
    textAlign: 'left',
  },
  featureCard: {
    backgroundColor: '#FFFFFF',
    padding: 20,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
    marginBottom: 16,
  },
  featureCardLarge: {
    padding: 32,
    borderRadius: 16,
  },
  featureHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  featureTitle: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginLeft: 8,
  },
  featureTitleLarge: {
    fontSize: 20,
    marginLeft: 12,
  },
  featureContent: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    lineHeight: 20,
  },
  featureContentLarge: {
    fontSize: 16,
    lineHeight: 24,
  },
  tipCard: {
    backgroundColor: '#FFFFFF',
    padding: 20,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  tipCardLarge: {
    padding: 32,
    borderRadius: 16,
  },
  tipHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  tipTitle: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginLeft: 8,
  },
  tipTitleLarge: {
    fontSize: 20,
    marginLeft: 12,
  },
  tipContent: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    lineHeight: 20,
  },
  tipContentLarge: {
    fontSize: 16,
    lineHeight: 24,
  },
});